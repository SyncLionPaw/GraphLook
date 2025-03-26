import os
from abc import ABC, abstractmethod
from typing import Optional

import aiofiles


from ..config import WORKSPACE_ROOT


class Function:
    def __init__(self, name, arguements):
        self.name: str = name
        self.arguments: str = arguements

    def to_dict(self):
        return {"name": self.name, "arguments": self.arguments}


class ToolCall:
    """Represents a tool/function call in a message"""

    def __init__(self, id, function):
        self.id: str = id
        self.type: str = "function"
        self.function: Function = function

    def to_dict(self):
        return {"id": self.id, "type": self.type, "function": self.function.to_dict()}


class BaseTool(ABC):
    def __init__(self, name, desc, params):
        self.name = name
        self.desc = desc
        self.params: Optional[dict] = params

    async def __call__(self, **kwargs):
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **params):
        pass

    # function_call 的形式
    def to_param(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.desc,
                "parameters": self.params,
            },
        }


class FileSaver(BaseTool):
    def __init__(self):
        name = "file_saver"
        desc = """Save content to a local file at a specified path.
            Use this tool when you need to save text, code, or generated content to a file on the local filesystem.
            The tool accepts content and a file path, and saves the content to that location.
        """
        params: dict = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "(required) The content to save to the file.",
                },
                "file_path": {
                    "type": "string",
                    "description": "(required) The path where the file should be saved, including filename and extension.",
                },
                "mode": {
                    "type": "string",
                    "description": "(optional) The file opening mode. Default is 'w' for write. Use 'a' for append.",
                    "enum": ["w", "a"],
                    "default": "w",
                },
            },
            "required": ["content", "file_path"],
        }
        super().__init__(name=name, desc=desc, params=params)

    async def execute(self, content: str, file_path: str, mode: str = "w") -> str:
        """保存文件内容 content 到指定的路径 file_path"""
        try:
            if os.path.isabs(file_path):
                file_name = os.path.basename(file_path)
                full_path = os.path.jpin(WORKSPACE_ROOT, file_name)
            else:
                full_path = os.path.join(WORKSPACE_ROOT, file_path)
            # Ensure the directory exists
            directory = os.path.dirname(full_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            async with aiofiles.open(full_path, mode, encoding="utf-8") as file:
                await file.write(content)
            return f"content successfully saved {full_path}"

        except Exception as e:
            return f"保存文件错误{str(e)}"
