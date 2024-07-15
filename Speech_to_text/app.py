import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import time
from speech_2_text import speech_2_text, downsampleWav
from config_app.config import get_config

config_app = get_config()
app = FastAPI()

@app.post("/process_voice/")
async def process_voice(data_type: UploadFile = File(...)):
    results = {"status": None, "message": "", "result": None}
    t1 = time.time()

    try:
        # Read the uploaded file
        data = await data_type.read()

        # Save the uploaded file
        src_output = './test_voice/test.wav'
        with open(src_output, "wb") as file_object:
            file_object.write(data)

        # Downsample the WAV file
        downsampleWav(src_output, src_output)

        # Process the audio file
        result = speech_2_text(src_output)
        results.update({
            "status": 200,
            "message": "Phát hiện được giọng nói.",
            "result": result
        })
    except Exception as e:
        results.update({
            "status": 500,
            "message": f"Error: {str(e)}"
        })

    # Return the results
    results['time_processing'] = time.time() - t1
    print('results:', results)
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config_app['server']['port'])
