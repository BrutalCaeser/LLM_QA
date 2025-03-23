from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
import torch
from src.rag.vector_db import VectorDB
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig,AutoConfig
#from collections import defaultdict
#from langchain.schema import BaseRetriever, Document
from typing import List, Any, Dict
from langchain.prompts import PromptTemplate
#from pydantic import Field  # Add this import
import logging
#"""DEFAULT_PROMPT = """Use the following pieces of context to answer the question at the end.
#If you don't know the answer, just say that you don't know, don't try to make up an answer.

#{context}

#Question: {question}
#Helpful Answer:
#- If the question asks for an average, calculate it from the context.
#- If the question asks for a list, filter by the specified criteria (e.g., country, nights).
#- If no matching data is found, say "No data found."



offload_folder = "D:/huggingface_offload"

class BookingQA:
    def __init__(self):
        self.vector_db = VectorDB()
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()
        self.qa_chain = self._create_chain()
        logger = logging.getLogger(__name__)

    def _init_retriever(self):
        """Initialize the retriever from vector store"""
        vector_store = self.vector_db.get_vector_store()
        return vector_store.as_retriever(search_kwargs={"k": 3})


    def _init_llm(self):
         """Initialize Mistral-7B with authentication and quantization"""
         #model_id = "google/gemma-3-1b-it"  # Ensure correct version
         #model_id="mistralai/Mistral-7B-Instruct-v0.1"
         model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
         hf_token = os.getenv("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")  # Store token in env variable

         quant_config = BitsAndBytesConfig(
             load_in_4bit=True,
             bnb_4bit_quant_type="nf4",
             bnb_4bit_compute_dtype=torch.float16,
         )

        # Load tokenizer and model with authentication
         tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token, trust_remote_code=True,padding_side="left",eos_token="<|endoftext|>",
            pad_token="<|pad|>")

         model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_auth_token=hf_token,
            trust_remote_code=True,
            #attn_impl='flash',# added - flash attention is faster
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
            offload_folder=offload_folder,

        )
         #model.tie_weights()
        # Use pipeline to create HuggingFacePipeline

         text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            #gpu_layers=35,
            max_new_tokens=128,
            #ffload_buffers=True,
            #load_in_4bit=True,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,  # Critical
        )

          # this line makes sure weight tying for less memory usage

         return text_gen_pipeline
    def _create_chain(self):
        #prompt_template = """Answer based on hotel booking context:
         #       {context}
          #      Question: {question}
           #     Answer:"""
        #"""Create retrieval QA chain"""
        #vector_store = self.vector_db.get_vector_store()
        # Create custom retriever
        #retriever = UniqueDocumentRetriever(vector_store, k=4)

        llm = HuggingFacePipeline(pipeline=self.llm)
        # ===== CUSTOM PROMPT SETUP =====

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            #retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True)
            #chain_type_kwargs = {
            #"prompt": PromptTemplate(
               # template=prompt_template,
                #input_variables=["context", "question"])})

    def ask(self, question: str,max_results: int=3) -> Dict[str, Any]:
        """Execute QA query with error handling"""
        try:
            # Update search parameters dynamically
            self.retriever.search_kwargs["k"] = max_results

            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except KeyError as e:
            logging.error(f"Missing key in response: {str(e)}")
            return {"error": "Response format mismatch"}
        except Exception as e:
            logging.error(f"QA failed: {str(e)}")
            return {"error": str(e)}
        #return self.qa_chain.invoke({"query": question}
# Quick test in Python console
#qa = BookingQA()
#print(hasattr(qa, 'retriever'))  # Should output: True
#response = qa.ask("What is the average lead time for IND country with nights greater than 3?")
#print(response["answer"])  # Should show numeric value
#print(len(response["source_documents"]))  # Should match max_results

"""class UniqueDocumentRetriever(BaseRetriever):
    # Explicitly declare fields for Pydantic validation
    vector_store: Any = Field(...)  # Required field
    k: int = Field(default=3)  # Optional field with default value

    def get_relevant_documents(self, query: str) -> List[Document]:
            # Retrieve documents from FAISS
        docs = self.vector_store.similarity_search(query, k=self.k)

            # Filter duplicates
        unique_docs = []
        seen = defaultdict(int)
        for doc in docs:
            content_hash = hash(doc.page_content)
            if seen[content_hash] == 0:
                unique_docs.append(doc)
                seen[content_hash] += 1

            # Query-specific filtering
        if "average lead time" in query.lower():
                # Filter outliers (e.g., lead time > 365 days)
            filtered_docs = [
                doc for doc in unique_docs
                if int(doc.page_content.split("Lead Time: ")[1].split(" ")[0]) <= 365
                ]
        elif "cancellation rate" in query.lower():
                # Filter by cancelled bookings
            filtered_docs = [
                doc for doc in unique_docs
                if "Cancelled: Yes" in doc.page_content
                ]
        elif "germany" in query.lower() and "nights" in query.lower():
                # Filter by country and nights
            filtered_docs = [
                doc for doc in unique_docs
                if "Country: DEU" in doc.page_content and
                    int(doc.page_content.split("Nights: ")[1].split(",")[0]) > 7
                ]
        else:
            filtered_docs = unique_docs

        return filtered_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
            # Async version (optional)
        return self.get_relevant_documents(query)"""


