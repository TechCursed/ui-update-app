from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import mysql.connector
import logging
import re
import time

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

# Database connection details
DB_USER = 'root'
DB_PASSWORD = 'admin'
DB_HOST = 'localhost'
DB_PORT = 3306

# Database connection functions
def get_mysql_schemas(user, password, host='localhost', port=3306):
    try:
        connection = mysql.connector.connect(user=user, password=password, host=host, port=port)
        cursor = connection.cursor()
        cursor.execute("SHOW DATABASES;")
        
        schemas = cursor.fetchall()
        cursor.close()
        connection.close()
        return [schema[0] for schema in schemas]
    except mysql.connector.Error as error:
        print(f"Error: {error}")
        return []

def get_schema_tables_info(user, password, schema, host='localhost', port=3306):
    try:
        connection = mysql.connector.connect(user=user, password=password, host=host, port=port, database=schema)
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        schema_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SHOW COLUMNS FROM {table_name};")
            columns = cursor.fetchall()
            column_info = [{'Field': col[0], 'Type': col[1], 'Null': col[2], 'Key': col[3], 'Default': col[4], 'Extra': col[5]} for col in columns]
            schema_info[table_name] = column_info
        cursor.close()
        connection.close()
        return schema_info
    except mysql.connector.Error as error:
        print(f"Error: {error}")
        return {}

def execute_query(user, password, schema, query, host='localhost', port=3306):
    try:
        connection = mysql.connector.connect(user=user, password=password, host=host, port=port, database=schema)
        cursor = connection.cursor()
        cursor.execute(query)
        column_names = tuple(i[0] for i in cursor.description)
        # column_names = [i[0] for i in cursor.description]
        logging.debug(column_names)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return column_names,result
    except mysql.connector.Error as error:
        print(f"Error: {error}")
        return (),[]

# LLM setup
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}
cache_dir = "./cache"
model_precision = "FP16"
inference_device = "GPU"
model_vendor = "meta-llama"
model_name = "Meta-Llama-3-8B"
model_id = f'{model_vendor}/{model_name}'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
ov_model_path = f'./{model_name}/{model_precision}'
model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config, cache_dir=cache_dir)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
llm = HuggingFacePipeline(pipeline=pipe)
prompt_template = PromptTemplate.from_template(
    """The database schema given below has schemas of different tables. Write a SQL query that would answer the user's QUESTION. Use the Schema Details to generate the SQL query. Please add SEMICOLON (;) at the end for the generated SQL query.
    {schema_representation}

    Question: {question}
    SQL Query:"""
)

def extract_sql(query):
    pattern = r"SQL Query:\s+(.*?);"
    result = re.search(pattern, query)
    if result:
        return result.group(1).strip()
    else:
        return "No match found"

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    try:
        schemas = get_mysql_schemas(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
        if not schemas:
            raise HTTPException(status_code=500, detail="No schemas found in the database.")

        schema_name = schemas[0]  # Assuming the first schema is the one we want to use
        schema_info = get_schema_tables_info(DB_USER, DB_PASSWORD, schema_name, DB_HOST, DB_PORT)

        schema_representation = f"Schema Information for {schema_name}:\n"
        for table, columns in schema_info.items():
            schema_representation += f"Table: {table}\n"
            for column in columns:
                schema_representation += (
                    f"  Column: {column['Field']}, Type: {column['Type']}, Null: {column['Null']}, "
                    f"Key: {column['Key']}, Default: {column['Default']}, Extra: {column['Extra']}\n"
                )
        chain = prompt_template | llm
        start_time = time.time()
        response = chain.invoke({"question": request.question, "schema_representation": schema_representation})
        logging.debug(response)
        query = extract_sql(response) + ";"
        column_names,result = execute_query(DB_USER, DB_PASSWORD, schema_name, query, DB_HOST, DB_PORT)
        output_res = [column_names] + result
        end_time = time.time()
        return {
            "query": query,
            "result": output_res,
            "time_taken": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
