import json
import logging
import sys

from fastapi import FastAPI, File, Response, UploadFile
from api.datatypes import CodeItem
from api.transcoder_client import TranscoderClient
from typing import Optional

app = FastAPI(
    title = "TransCoder API",
    description = "An unsupervised programming language translator.",
    version = "0.0.1"
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@app.get("/hello")
def get_root():
    return {"Hello from TransCoder API!"}

@app.post("/translate")
async def post_translate(code_item: CodeItem):
    code_str = json.loads(code_item.code_str)
    logging.info(f"INPUT: {code_str}")

    output = await get_output(
        code_str=code_str,
        src_lang=code_item.src_lang,
        tgt_lang=code_item.tgt_lang,
        beam_size=code_item.beam_size
    )

    logging.info(f"OUTPUT: {output}")
    return Response(content=json.dumps({"translation": output}))

@app.post("/translate_file")
async def post_translate_file(src_lang: str, tgt_lang: str, code_file: UploadFile = File(...), beam_size: Optional[int] = 1):
    file_contents = code_file.file.read().decode("utf-8").strip()
    logging.info(f"INPUT: {file_contents}")

    output = await get_output(
        code_str=file_contents,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        beam_size=beam_size
    )

    logging.info(f"OUTPUT: {output}")
    return Response(content=json.dumps({"translation": output}))

async def get_output(code_str: str, src_lang: str, tgt_lang: str, beam_size: int):
    translator = TranscoderClient(src_lang, tgt_lang)
    return translator.translate(code_str, lang1=src_lang, lang2=tgt_lang, beam_size=beam_size)
