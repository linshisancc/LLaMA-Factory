import json
import os
from typing import List, Tuple
from pydantic import BaseModel
from contextlib import asynccontextmanager

from llmtuner.api.protocol import (
    Role,
    Finish,
    ModelCard,
    ModelList,
    ChatMessage,
    DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseUsage,
    ScoreEvaluationRequest,
    ScoreEvaluationResponse
)
from llmtuner.chat import ChatModel
from llmtuner.extras.misc import torch_gc
from llmtuner.extras.packages import (
    is_fastapi_availble, is_starlette_available, is_uvicorn_available
)

import tiktoken
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sentence_transformers import SentenceTransformer


if is_fastapi_availble():
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware


if is_starlette_available():
    from sse_starlette import EventSourceResponse


if is_uvicorn_available():
    import uvicorn

# m3e 向量模型
M3E_MODEL_PATH = os.environ.get('M3E_MODEL_PATH', '/home/linshisancc/models/m3e-base')

@asynccontextmanager
async def lifespan(app: "FastAPI"): # collects GPU memory
    yield
    torch_gc()


def to_json(data: BaseModel) -> str:
    try: # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except: # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str

class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

def expand_features(embedding, target_length):
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
    expanded_embedding = expanded_embedding.flatten()
    if len(expanded_embedding) > target_length:
        # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
        expanded_embedding = expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
        expanded_embedding = np.pad(
            expanded_embedding, (0, target_length - len(expanded_embedding))
        )
    return expanded_embedding

embeddings_model = SentenceTransformer(M3E_MODEL_PATH, device='cpu')

def create_app(chat_model: "ChatModel") -> "FastAPI":
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 增加 embeddings 接口 用于 fastgpt 知识库
    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def get_embeddings(
        request: EmbeddingRequest
    ):
        # 计算嵌入向量和tokens数量
        embeddings = [embeddings_model.encode(text) for text in request.input]

        # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度
        embeddings = [
            expand_features(embedding, 1536) if len(embedding) < 1536 else embedding
            for embedding in embeddings
        ]

        # Min-Max normalization 归一化
        embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

        # 将numpy数组转换为列表
        embeddings = [embedding.tolist() for embedding in embeddings]
        prompt_tokens = sum(len(text.split()) for text in request.input)
        total_tokens = sum(num_tokens_from_string(text) for text in request.input)

        response = {
            "data": [
                {"embedding": embedding, "index": index, "object": "embedding"}
                for index, embedding in enumerate(embeddings)
            ],
            "model": request.model,
            "object": "list",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            },
        }

        return response

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion(request: ChatCompletionRequest):
        if not chat_model.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        if len(request.messages) == 0 or request.messages[-1].role != Role.USER:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request")

        query = request.messages[-1].content
        prev_messages = request.messages[:-1]
        if len(prev_messages) and prev_messages[0].role == Role.SYSTEM:
            system = prev_messages.pop(0).content
        else:
            system = None

        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == Role.USER and prev_messages[i+1].role == Role.ASSISTANT:
                    history.append([prev_messages[i].content, prev_messages[i+1].content])
                else:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")

        if request.stream:
            generate = predict(query, history, system, request)
            return EventSourceResponse(generate, media_type="text/event-stream")

        responses = chat_model.chat(
            query, history, system,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            num_return_sequences=request.n
        )

        prompt_length, response_length = 0, 0
        choices = []
        for i, response in enumerate(responses):
            choices.append(ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(role=Role.ASSISTANT, content=response.response_text),
                finish_reason=Finish.STOP if response.finish_reason == "stop" else Finish.LENGTH
            ))
            prompt_length = response.prompt_length
            response_length += response.response_length

        usage = ChatCompletionResponseUsage(
            prompt_tokens=prompt_length,
            completion_tokens=response_length,
            total_tokens=prompt_length+response_length
        )

        return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)

    async def predict(query: str, history: List[Tuple[str, str]], system: str, request: ChatCompletionRequest):
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role=Role.ASSISTANT, content=""),
            finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield to_json(chunk)

        for new_text in chat_model.stream_chat(
            query, history, system,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens
        ):
            if len(new_text) == 0:
                continue

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
            yield to_json(chunk)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason=Finish.STOP
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield to_json(chunk)
        yield "[DONE]"

    @app.post("/v1/score/evaluation", response_model=ScoreEvaluationResponse, status_code=status.HTTP_200_OK)
    async def create_score_evaluation(request: ScoreEvaluationRequest):
        if chat_model.can_generate:
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        if len(request.messages) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request")
        
        scores = chat_model.get_scores(request.messages, max_length=request.max_length)
        return ScoreEvaluationResponse(model=request.model, scores=scores)

    return app


if __name__ == "__main__":
    chat_model = ChatModel()
    app = create_app(chat_model)
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
