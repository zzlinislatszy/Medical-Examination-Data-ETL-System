from fastapi import FastAPI
from text_processing_251029 import router

app = FastAPI(title="Text Processing Pipeline Demo API", version="1.0.0")
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Text Processing Pipeline Demo API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
