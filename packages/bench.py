import os
import json
import chromadb
from splitter.FixedSizeFixedStepSplitter import FiexedSizeFixedStepSplitter
from splitter.DynamicSizeFixedStepSplitter import DynamicSizeFixedStepSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel

output_folder = "packages/rag-ollama-multi-query/data/multifieldqa_en_bench/multifieldqa_en_answer_DFS_512_256"
input_folder = "packages/rag-ollama-multi-query/data/multifieldqa_en"
os.makedirs(output_folder, exist_ok=True)  # 创建文件夹，如果已存在则不会报错

for i in range(0, len(os.listdir(input_folder))):
    print(i)
    input_file = f"{input_folder}/output_{i}.json"
    output_file = f"{output_folder}/aoutput_{i}.json"

    with open(input_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)

        # 创建数据库客户端
        client = chromadb.Client()
        if i != 0:
            client.delete_collection("rag-private")

        # 分割文本
        text_splitter = DynamicSizeFixedStepSplitter(chunk_size=512, step_window=256)
        all_splits = text_splitter.split_text(json_data["context"])

        # 将文本向量化并存储到 Chroma 中
        vectorstore = Chroma.from_texts(
            texts=all_splits,
            collection_name="rag-private",
            embedding=OllamaEmbeddings(),
        )
        retriever = vectorstore.as_retriever()

        # 设置 RAG 模型
        prompt_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        ollama_llm = "llama3"

        model = ChatOllama(model = ollama_llm)
        chain = (
                RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
                | prompt
                | model
                | StrOutputParser()
        )

        # 为输入添加类型
        class Question(BaseModel):
            __root__: str

        chain = chain.with_types(input_type=Question)

        # 运行链式流程并保存结果到输出文件
        output = {}
        try:
            output["answer"] = chain.invoke(json_data["input"])
            print(json_data["answers"])
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(output, file, ensure_ascii=False, indent=4)

        except Exception as e:
            print("Error:", e)
