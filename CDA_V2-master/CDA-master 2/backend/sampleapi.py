from fastapi import FastAPI
from typing import List, Union
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins. For production, specify the allowed origins.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# CORS(app)
class Result(BaseModel):
    date: str
    description: str
    type: str
    amount: Union[float, int]

class QueryResponse(BaseModel):
    query: str
    result: List[Result]
    time_taken: float

class QueryRequest(BaseModel):
    parameter: str

@app.post("/ask_question", response_model=QueryResponse)
def get_query_result(request: QueryRequest):
    # You can use the parameter from the request to influence the response
    parameter = request.parameter
    
    # Dummy data - in a real application, this would be dynamically generated or retrieved
    data = {
        "query": f"select * from book where parameter = '{parameter}';",
        "result": [
            ["30-12-2023", "Belastingdienst", "Expense", 9.96],
            ["30-12-2023", "Tesco Breda", "Expense", 17.53],
            ["30-12-2023", "Monthly Appartment Rent", "Expense", 451],
            ["30-12-2023", "Vishandel Sier Amsterdam", "Expense", 12.46],
            ["29-12-2023", "Selling Paintings", "Income", 13.63],
            ["29-12-2023", "Spotify Ab By Adyen", "Expense", 12.19],
            ["23-12-2023", "Tk Maxx Amsterdam Da", "Expense", 27.08],
            ["22-12-2023", "Consulting", "Income", 541.57],
            ["22-12-2023", "Aidsfonds", "Expense", 10.7],
            ["20-12-2023", "Consulting", "Income", 2641.93],
            ["19-12-2023", "Tls Bv Inz Ov-Chipkaart", "Expense", 18.9],
            ["18-12-2023", "Etos Amsterdam", "Expense", 17.67],
            ["18-12-2023", "Tesco Breda", "Expense", 8.81],
            ["18-12-2023", "Beta Boulders Ams Amsterdam", "Expense", 6.94],
            ["26-11-2022", "Salary", "Income", 14.36],
            ["26-11-2022", "Bouldermuur Bv Amsterdam", "Expense", 19.27],
            ["26-11-2022", "Birtat Restaurant Amsterdam", "Expense", 24.71],
            ["25-11-2022", "Tesco Breda", "Expense", 17.35],
            ["24-11-2022", "Freelancing", "Income", 2409.55],
            ["19-11-2022", "Tikkie", "Expense", 20.76],
            ["25-10-2022", "Blogging", "Income", 4044.27],
            ["24-10-2022", "Taxi Utrecht", "Expense", 18.9],
            ["23-10-2022", "Tesco Breda", "Expense", 27.54],
            ["22-10-2022", "Apple Services", "Expense", 41.25],
            ["21-10-2022", "Tesco Breda", "Expense", 22.8],
            ["16-01-2022", "Amazon Lux", "Expense", 24.11]
        ],
        "time_taken": 67.53975629806519
    }

    # Convert the result list to Result objects
    result_objects = [Result(date=item[0], description=item[1], type=item[2], amount=item[3]) for item in data["result"]]

    return QueryResponse(
        query=data["query"],
        result=result_objects,
        time_taken=data["time_taken"]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
