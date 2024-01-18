# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess
import json
 
MODELS_PATH = "model_base/"
 
app = FastAPI()

class ShellScriptRequest(BaseModel):
    script_path: str
    preprocessed_file: str
    natsqltables_file: str
    output_file: str



 
 
@app.get('/ping')
async def ping():
    return {"message": "ok"}
 
@app.on_event('startup')
def load_model():
    # classifier = pipeline("ner", model=MODElS_PATHA)
    # logging.info("Model loaded.")
    return "classifier"
 
@app.post('/invocations')
def invocations(request: ShellScriptRequest):
    print("Request",request)
    try:
        # Extract values from the request
        script_path = request.script_path
        preprcessed_file = request.preprocessed_file
        natsqltables_file = request.natsqltables_file
        output_file = request.output_file

        # Build the command
        command = [
            "sh",
            script_path,
            preprcessed_file,
            natsqltables_file,
            output_file
        ]

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        print("Results: ", result)

        if result.returncode == 0:
            with open(preprcessed_file, 'r', encoding="utf-8") as json_file:
                preprocessed_data = json.load(json_file)
            user_question = [item['question'] for item in preprocessed_data]
            with open(output_file, 'r') as sql_file:
                sql_query = sql_file.read()
            response_data = {"UserQuestion": user_question[0], "Prediction": sql_query}
            return JSONResponse(content=response_data, status_code=200)
        else:
            raise HTTPException(status_code=500, detail=f"An error occurred: {result.stderr}")

    except Exception as e:
        # Handle exceptions and return an appropriate response
        error_message = f"An error occurred: {str(e)}"
        return HTTPException(status_code=500, detail=error_message)

    #return {"message": "Invocation Request Received"}
