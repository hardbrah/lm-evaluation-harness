"""
基于 YAML 配置驱动的 log_sample 处理脚本

使用方法:
    python hf_template.py                           # 使用默认配置文件 process_config.yaml
    python hf_template.py --config my_config.yaml   # 使用指定配置文件

功能：
1. 从 log_sample (jsonl) 中解析 response
2. 根据阈值对 response 进行截断（整数=token数，浮点0-1=百分比）
3. 从 doc 中提取 question，使用 jinja2 模板拼接提示词
4. 转换成 chat 格式，使用 apply_chat_template 获取最终 string
"""

import argparse
import json
from pathlib import Path
from typing import Union, List, Dict, Any
from dataclasses import dataclass, field

import yaml
from transformers import AutoTokenizer
from jinja2 import Template


@dataclass
class ProcessConfig:
    """处理配置（从 YAML 加载）"""
    # 数据路径
    log_sample_path: str = ""
    output_path: str = ""
    
    # 模型配置
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    
    # 截断阈值：整数表示 token 数，浮点数 (0-1) 表示百分比
    truncate_threshold: Union[int, float] = 1.0
    
    # response 选择：选择第几个 response（从 0 开始），-1 表示所有
    response_index: int = 0
    
    # doc_to_text 的 jinja2 模板
    doc_to_text_template: str = "{{ question }}"
    
    # 输出结果新增的 key 名称
    output_key: str = "text"
    
    # 是否启用 thinking 模式（Qwen3 non-thinking 模型应设为 False）
    enable_thinking: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ProcessConfig":
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 过滤掉 None 值和不存在的字段
        valid_fields = {k: v for k, v in config_dict.items() 
                       if v is not None and hasattr(cls, k.__str__()) or k in cls.__dataclass_fields__}
        
        return cls(**valid_fields)
    
    def to_yaml(self, yaml_path: str):
        """保存配置到 YAML 文件"""
        config_dict = {
            'log_sample_path': self.log_sample_path,
            'output_path': self.output_path,
            'model_name_or_path': self.model_name_or_path,
            'truncate_threshold': self.truncate_threshold,
            'response_index': self.response_index,
            'doc_to_text_template': self.doc_to_text_template,
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)


class LogSampleProcessor:
    """Log Sample 处理器"""
    
    def __init__(self, config: ProcessConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.doc_to_text_renderer = Template(config.doc_to_text_template)
    
    def load_samples(self) -> List[Dict[str, Any]]:
        """从 jsonl 文件加载 samples"""
        samples = []
        with open(self.config.log_sample_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    
    def extract_response(self, sample: Dict[str, Any]) -> Union[str, List[str]]:
        """从 sample 中提取 response"""
        resps = sample.get("resps", [[]])
        if not resps or not resps[0]:
            return "" if self.config.response_index >= 0 else []
        
        responses = resps[0]  # 第一组 responses
        
        if self.config.response_index >= 0:
            if self.config.response_index < len(responses):
                return responses[self.config.response_index]
            return ""
        else:
            return responses
    
    def truncate_response(self, response: str) -> str:
        """根据阈值截断 response
        
        Args:
            response: 原始 response 文本
            
        Returns:
            截断后的 response
        """
        threshold = self.config.truncate_threshold
        
        # 如果阈值为 1.0（浮点）或小于等于 0，不截断
        if isinstance(threshold, float) and threshold >= 1.0:
            return response
        if isinstance(threshold, int) and threshold <= 0:
            return response
        
        # 对 response 进行 tokenize
        tokens = self.tokenizer.encode(response, add_special_tokens=False)
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return response
        
        # 计算截断位置
        if isinstance(threshold, float) and 0 < threshold < 1:
            # 浮点数：按百分比截断
            keep_tokens = int(total_tokens * threshold)
        elif isinstance(threshold, int) and threshold > 0:
            # 整数：按 token 数截断
            keep_tokens = min(threshold, total_tokens)
        else:
            return response
        
        # 截断并解码
        truncated_tokens = tokens[:keep_tokens]
        truncated_response = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_response
    
    def render_doc_to_text(self, doc: Dict[str, Any]) -> str:
        """使用 jinja2 模板渲染 doc_to_text"""
        question = doc.get("question", "")
        return self.doc_to_text_renderer.render(question=question, doc=doc)
    
    def build_chat_messages(self, user_content: str, assistant_content: str) -> List[Dict[str, str]]:
        """构建 chat 格式的 messages"""
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """应用 chat template 获取格式化的字符串
        
        关键参数说明：
        - tokenize=False: 返回字符串而非 token ids
        - add_generation_prompt=False: 不添加额外的生成提示
        - continue_final_message=True: 继续最后一条消息（不添加结束标记）
        - enable_thinking: 是否启用 thinking 模式（Qwen3 non-thinking 模型应设为 False）
        """
        # 构建基本参数
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
            "continue_final_message": True,
        }
        
        # 尝试添加 enable_thinking 参数（Qwen3 系列模型支持）
        # non-thinking 模型应设为 False 以避免 <think> 标签
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                enable_thinking=self.config.enable_thinking,
                **kwargs
            )
        except TypeError:
            # 如果模型不支持 enable_thinking 参数，则不使用
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        
        # 如果 enable_thinking=False 但输出仍包含 <think> 标签，手动移除
        if not self.config.enable_thinking and "<think>" in prompt:
            import re
            prompt = re.sub(r'<think>\s*</think>\s*', '', prompt)
        
        return prompt
    
    def process_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理单个 sample，为每个 response 生成独立记录
        
        如果一个 doc 有 N 个 response，则拆分成 N 条记录。
        每条记录包含：doc_id, doc, arguments, response（单个）, text（格式化文本）
        
        Returns:
            记录列表，每个 response 对应一条记录
        """
        doc = sample.get("doc", {})
        doc_id = sample.get("doc_id")
        arguments = sample.get("arguments", {})
        
        # 获取所有 responses
        resps = sample.get("resps", [[]])
        all_responses = resps[0] if resps and resps[0] else []
        
        # 如果指定了特定索引，只处理那一个
        if self.config.response_index >= 0:
            if self.config.response_index < len(all_responses):
                all_responses = [all_responses[self.config.response_index]]
            else:
                all_responses = []
        
        # 渲染 doc_to_text（对所有 response 相同）
        user_content = self.render_doc_to_text(doc)
        
        results = []
        for idx, response in enumerate(all_responses):
            # 截断 response
            truncated_response = self.truncate_response(response)
            
            # 构建 chat messages
            messages = self.build_chat_messages(user_content, truncated_response)
            
            # 应用 chat template
            formatted_text = self.apply_chat_template(messages)
            
            # 构建输出记录（精简字段，符合数据集要求）
            result = {
                "doc_id": doc_id,
                "doc": doc,
                "arguments": arguments,
                "response": response,  # 单个原始 response
                "response_idx": idx,   # response 索引
                self.config.output_key: formatted_text,  # 格式化后的 text
            }
            results.append(result)
        
        return results
    
    def process_all(self) -> List[Dict[str, Any]]:
        """处理所有 samples，展开多个 response"""
        samples = self.load_samples()
        results = []
        for sample in samples:
            sample_results = self.process_sample(sample)
            results.extend(sample_results)  # 展开列表
        return results
    
    # 默认输出目录（当 output_path 留空时使用）
    DEFAULT_OUTPUT_DIR = "./results"

    def save_results(self, results: List[Dict[str, Any]], output_path: str = None):
        """保存处理结果到 jsonl 文件。output_path 留空时保存到默认路径 ./results/"""
        raw_path = output_path or self.config.output_path or self.DEFAULT_OUTPUT_DIR
        path = Path(raw_path)
        
        # 如果路径是目录或不带扩展名，自动生成文件名
        if path.is_dir() or (path.exists() and path.is_dir()) or not path.suffix:
            # 基于输入文件名生成输出文件名
            input_name = Path(self.config.log_sample_path).stem
            output_filename = f"{input_name}_processed.jsonl"
            
            # 确保目录存在
            path.mkdir(parents=True, exist_ok=True)
            path = path / output_filename
        else:
            # 如果输出目录不存在，自动创建
            output_dir = path.parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"已创建输出目录: {output_dir}")
        
        with open(path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"结果已保存到: {path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于 YAML 配置驱动的 log_sample 处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python hf_template.py                           # 使用默认配置
    python hf_template.py --config my_config.yaml   # 使用自定义配置
    python hf_template.py --show-sample             # 仅显示示例结果
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML 配置文件路径（默认: 同目录下的 process_config.yaml）"
    )
    parser.add_argument(
        "--show-sample", "-s",
        action="store_true",
        help="仅显示第一个样本的处理结果"
    )
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    
    # 确定配置文件路径
    if args.config:
        config_path = args.config
    else:
        # 默认使用同目录下的 process_config.yaml
        script_dir = Path(__file__).parent
        config_path = script_dir / "process_config.yaml"
    
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        print("请创建配置文件或使用 --config 指定路径")
        return 1
    
    print(f"加载配置: {config_path}")
    
    # 加载配置
    config = ProcessConfig.from_yaml(config_path)
    
    print(f"处理文件: {config.log_sample_path}")
    print(f"模型: {config.model_name_or_path}")
    print(f"截断阈值: {config.truncate_threshold}")
    print(f"输出 key: {config.output_key}")
    print(f"enable_thinking: {config.enable_thinking}")
    print(f"提示词模板: {config.doc_to_text_template[:50]}...")
    print()
    
    # 处理
    processor = LogSampleProcessor(config)
    results = processor.process_all()
    
    print(f"处理完成，共 {len(results)} 个样本")
    
    # 保存结果（路径留空时保存到默认 ./results/）
    processor.save_results(results)
    
    # 显示示例结果
    if args.show_sample:
        if results:
            print("\n" + "=" * 80)
            print("示例处理结果 (第一条记录):")
            print("=" * 80)
            
            result = results[0]
            output_key = config.output_key
            
            # 显示字段
            print(f"\n[Doc ID]: {result.get('doc_id', 'N/A')}")
            print(f"[Response IDX]: {result.get('response_idx', 'N/A')}")
            
            doc = result.get('doc', {})
            question = doc.get('question', 'N/A')
            print(f"\n[Question]:\n{question[:200]}{'...' if len(question) > 200 else ''}")
            
            response = result.get('response', '')
            print(f"\n[Response] (原始，{len(response)} chars):\n{response[:200]}{'...' if len(response) > 200 else ''}")
            
            formatted_text = result.get(output_key, '')
            print(f"\n[{output_key}] (格式化文本):\n{formatted_text}")
            
            print("\n" + "=" * 80)
            print(f"输出字段: {list(result.keys())}")
            print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
