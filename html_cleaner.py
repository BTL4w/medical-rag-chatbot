"""
YouMed content cleaner
Transforms content field in JSONL using custom rules.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterable
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_addison_logic(text: str) -> str:
    # 0. Bỏ phần text trước \n\n<h2> đầu tiên
    marker = '\n\n<h2>'
    if marker in text:
        text = text.split(marker, 1)[1]
        text = marker + text

    # 1. Xử lý Header (Mở + Đóng)
    # \n\n<h2> -> \n\n## (Cần regex để bắt pattern \n\n<h..>)
    text = text.replace('\n\n<h2>', '\n\n## ')
    text = text.replace('\n\n<h3>', '\n\n### ')
    
    # </h2>\n\n -> \n (Thu hẹp khoảng cách)
    text = text.replace('</h2>\n\n', '\n')
    text = text.replace('</h3>\n\n', '\n')
    
    # 2. Xử lý List
    # \n\n<li> -> \n* (Thu hẹp khoảng cách giữa các item và nối với đoạn dẫn)
    text = text.replace('\n\n<li>', '\n* ')
    
    # </li> -> Xóa (Dọn rác)
    text = text.replace('</li>', '')
    
    # Các \n\n còn lại tự động được giữ nguyên
    return text


class YouMedContentCleaner:
    """Clean content field from JSONL and write to processed output."""

    def __init__(self, input_file: str = "data/raw/youmed_articles_test.jsonl", output_dir: str = "data/processed"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / self.input_file.name

    def _iter_jsonl(self) -> Iterable[Dict[str, Any]]:
        with self.input_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def clean(self) -> None:
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        count = 0
        with self.output_file.open('w', encoding='utf-8') as out:
            for record in self._iter_jsonl():
                content = record.get('content', '')
                if isinstance(content, str) and content:
                    record['content'] = clean_addison_logic(content)
                json.dump(record, out, ensure_ascii=False)
                out.write('\n')
                count += 1

        logger.info(f"Cleaned {count} records to {self.output_file}")


if __name__ == "__main__":
    # Tìm tất cả các file youmed_articles_*.jsonl trong data/raw và clean từng file
    cleaner = YouMedContentCleaner()
    cleaner.clean()
    # input_files = glob("data/raw/youmed_articles_*.jsonl")
    # for input_file in input_files:
    #     cleaner = YouMedContentCleaner(input_file=input_file)
    #     cleaner.clean()
