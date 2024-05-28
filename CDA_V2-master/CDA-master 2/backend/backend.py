from fastapi import FastAPI, UploadFile, File
import csv
import mysql.connector
import io
import pandas as pd
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import tempfile
from tempfile import NamedTemporaryFile
from sqlalchemy import create_engine, Table, Column, MetaData, ForeignKey
from sqlalchemy.dialects.mysql import VARCHAR
import chardet
import numpy as np
from pydantic import BaseModel

import subprocess

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)

# Global variable to store the db_name
db_name_global = None
# Global variables to store file info
uploaded_files_info = []
temp_file_paths = []
file_names = []
dataVizFile = None

app = FastAPI()

# Allow CORS for your frontend origin
origins = [
    "http://127.0.0.1:5501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateDBRequest(BaseModel):
    db_name: str


def start_streamlit_app(data_viz_file):
    # Define the command to run the Streamlit app with the data file as an argument
    command = f"streamlit run visualization.py -- --dataVizFile {data_viz_file}"
    # Start the Streamlit app using subprocess
    subprocess.Popen(command, shell=True)


@app.post("/create_db")
async def create_db(request: CreateDBRequest):
    global db_name_global
    db_name_global = request.db_name

    connection = mysql.connector.connect(
        host="localhost", user="root", password="asdfgh@3"
    )
    cursor = connection.cursor()

    cursor.execute("SHOW DATABASES")
    databases = cursor.fetchall()

    if (db_name_global,) in databases:
        cursor.close()
        connection.close()
        return {
            "status": f"Database {db_name_global} already exists. Using existing database."
        }
    else:
        try:
            cursor.execute(f"CREATE DATABASE {db_name_global}")
            cursor.close()
            connection.close()
            return {"status": f"Database {db_name_global} created successfully."}
        except mysql.connector.Error as err:
            cursor.close()
            connection.close()
            return {"error": f"Error creating database: {err}"}


@app.post("/upload_file_info")
async def upload_file_info(files: List[UploadFile] = File(...)):
    global uploaded_files_info
    global file_names
    global dataVizFile
    uploaded_files_info = []
    file_infos = []
    file_names = []
    for file in files:
        file_extension = os.path.splitext(file.filename)[-1].lower()
        file_content = await file.read()  # Read file content
        file_names.append(
            file.filename.split(".")[0]
        )  # Store file name without extension
        # Detect file encoding
        result = chardet.detect(file_content)
        encoding = result["encoding"]
        # Load data into DataFrame based on file extension
        if file_extension == ".csv":
            df = pd.read_csv(io.StringIO(file_content.decode(encoding)))
        elif file_extension == ".xlsx":
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            return {
                "error": f"Invalid file format in {file.filename}. Please upload a CSV or XLSX file."
            }
        # Calculate file size
        file_size = len(file_content) / 1024 / 1024  # Size in MB
        # Store file info
        file_info = {
            "filename": file.filename,
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "file_size(MB)": file_size,
        }
        # Save the file content to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.write(file_content)
        temp_file.close()
        dataVizFile = temp_file.name
        print(
            "-------------------------",
            dataVizFile,
            "------------------------------------",
        )
        logging.debug("File name : ", dataVizFile)
        # Append the file path and name to the global list
        uploaded_files_info.append(
            {"file_path": temp_file.name, "file_name": file.filename}
        )
        # Append the file info to the list
        file_infos.append(file_info)
        logging.debug(uploaded_files_info)
    # Log the lengths of the lists
    logging.debug(f"File names after upload_file_info: {file_names}")
    logging.debug(f"Temporary file paths after upload_file_info: {uploaded_files_info}")
    logging.debug(f"Length of file_names after upload_file_info: {len(file_names)}")
    logging.debug(
        f"Length of uploaded_files_info after upload_file_info: {len(uploaded_files_info)}"
    )

    # Start the Streamlit app with the data file
    start_streamlit_app(dataVizFile)

    return {
        "file_info": file_infos,
        "saved_files": uploaded_files_info,
        "file_names": file_names,
        "dataVizFile": dataVizFile,
    }


@app.post("/upload_and_clean")
async def upload_and_clean():
    global uploaded_files_info
    global temp_file_paths  # Reference the global variable
    sanitization_infos = []

    for file_info in uploaded_files_info:
        file_path = file_info["file_path"]
        file_name = file_info["file_name"]
        file_extension = os.path.splitext(file_path)[-1].lower()

        with open(file_path, "rb") as file:
            file_content = file.read()

        # Detect file encoding
        result = chardet.detect(file_content)
        encoding = result["encoding"]

        # Load the DataFrame based on file extension
        if file_extension == ".csv":
            df = pd.read_csv(io.StringIO(file_content.decode(encoding)))
        elif file_extension == ".xlsx":
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            return {
                "error": f"Invalid file format in {file_name}. Please upload a CSV or XLSX file."
            }

        sanitization_info = {"filename": file_name}

        # Data cleaning
        sanitization_info["original_shape"] = df.shape

        # Convert column names to lower case and replace spaces with underscore
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        sanitization_info["column_names_sanitized"] = True

        # Replace special characters in column names
        df.columns = df.columns.str.replace(r"\W", "", regex=True)
        sanitization_info["special_characters_removed_from_column_names"] = True

        # Strip leading/trailing whitespace from column names and values
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        sanitization_info["whitespace_removed"] = True

        # Convert date columns to standard date formats
        for col in df.select_dtypes(include=["object"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception as e:
                pass
        sanitization_info["dates_standardized"] = True

        # Fill missing values with 'NA'
        initial_na_count = df.isna().sum().sum()
        df = df.fillna("NA")
        sanitization_info["missing_values_filled"] = int(initial_na_count)

        # Remove duplicate rows
        initial_duplicates_count = df.duplicated().sum()
        df = df.drop_duplicates()
        sanitization_info["duplicates_removed"] = int(initial_duplicates_count)

        # Save the cleaned DataFrame to a temporary file
        temp_file = NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file_path = temp_file.name
        if file_extension == ".csv":
            df.to_csv(temp_file_path, index=False)
        elif file_extension == ".xlsx":
            df.to_excel(temp_file_path, index=False)
        sanitization_info["temp_file_path"] = temp_file_path
        temp_file_paths.append(temp_file_path)
        logging.debug("-----------------------------------------------")
        logging.debug(temp_file_paths)
        sanitization_infos.append(sanitization_info)

    # Log the lengths of the lists
    logging.debug(f"Temporary file paths after cleaning: {temp_file_paths}")
    logging.debug(f"Length of temp_file_paths after cleaning: {len(temp_file_paths)}")
    logging.debug(f"Length of file_names after cleaning: {len(file_names)}")

    # Clear uploaded_files_info after processing
    uploaded_files_info.clear()

    return {
        "status": "Data cleaned and saved",
        "sanitization_infos": sanitization_infos,
    }


def infer_primary_key(df):
    for column in df.columns:
        if df[column].is_unique:
            return column
    return None


@app.post("/create_tables_with_relationships")
async def create_tables_with_relationships():
    global temp_file_paths
    global db_name_global
    global file_names

    # Log the lengths of the lists before creating tables
    logging.debug(f"Temporary file paths before creating tables: {temp_file_paths}")
    logging.debug(f"File names before creating tables: {file_names}")
    logging.debug(
        f"Length of temp_file_paths before creating tables: {len(temp_file_paths)}"
    )
    logging.debug(f"Length of file_names before creating tables: {len(file_names)}")

    if file_names is None:
        return {"error": "file_names is not initialized."}

    if len(temp_file_paths) != len(file_names):
        return {"error": "Mismatch between number of temporary files and file names."}

    engine = create_engine(
        f"mysql+mysqlconnector://root:12345@localhost/{db_name_global}"
    )
    metadata = MetaData()

    messages = []
    table_definitions = {}
    relationships = []

    for temp_file_path, file_name in zip(temp_file_paths, file_names):
        file_extension = os.path.splitext(temp_file_path)[-1].lower()

        if file_extension == ".csv":
            df = pd.read_csv(temp_file_path)
        elif file_extension == ".xlsx":
            df = pd.read_excel(temp_file_path)
        else:
            return {
                "error": f"Invalid file format in {temp_file_path}. Please upload a CSV or XLSX file."
            }

        table_name = file_name.lower().replace(" ", "_")

        primary_key_column = infer_primary_key(df)
        if not primary_key_column:
            messages.append(
                {
                    "file": temp_file_path,
                    "status": f"No unique column found for primary key in {file_name}",
                }
            )
            continue

        columns = [Column(col, VARCHAR(255)) for col in df.columns]
        for column in columns:
            if column.name == primary_key_column:
                column.primary_key = True
        table = Table(table_name, metadata, *columns)

        table_definitions[table_name] = table
        messages.append(
            {
                "file": temp_file_path,
                "status": f"Primary key {primary_key_column} identified for table {table_name}",
            }
        )

    for table_name, table in table_definitions.items():
        for column in table.columns:
            for ref_table_name in table_definitions.keys():
                if (
                    ref_table_name != table_name
                    and column.name == f"{ref_table_name}_id"
                ):
                    foreign_key = ForeignKey(f"{ref_table_name}.{column.name}")
                    relationships.append((table, column.name, foreign_key))

    metadata.create_all(engine)

    for table, column_name, foreign_key in relationships:
        with engine.connect() as conn:
            conn.execute(
                f"ALTER TABLE {table.name} ADD CONSTRAINT fk_{table.name}_{column_name} FOREIGN KEY ({column_name}) REFERENCES {foreign_key.target_fullname}"
            )

    for temp_file_path, file_name in zip(temp_file_paths, file_names):
        file_extension = os.path.splitext(temp_file_path)[-1].lower()
        table_name = file_name.lower().replace(" ", "_")

        if file_extension == ".csv":
            df = pd.read_csv(temp_file_path)
        elif file_extension == ".xlsx":
            df = pd.read_excel(temp_file_path)

        df.to_sql(table_name, con=engine, if_exists="append", index=False)
        messages.append(
            {"file": temp_file_path, "status": f"Data inserted into table {table_name}"}
        )

        os.remove(temp_file_path)
        messages.append({"file": temp_file_path, "status": "Temporary file deleted"})

    temp_file_paths.clear()
    file_names.clear()

    return {"status": "Data processing completed", "details": messages}
