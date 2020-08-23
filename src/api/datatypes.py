from enum import Enum
from pydantic import BaseModel
from typing import Optional


class Languages(Enum):
    CPP = 'cpp'
    JAVA = 'java'
    PYTHON = 'python'

class CodeItem(BaseModel):
    code_str: str
    src_lang: str
    tgt_lang: str
    beam_size: Optional[int] = 1
