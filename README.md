# Funcion Calling Benchmark for MPC conversation

## Environment

- python 3.10  
- requirements.txt 참조

## Running

python main.py  

### Arguments

- src_configs/gen_base.yaml 수정

fewshot이랑 function list를 txt/json file화해서 쉽게 import할 수 있게 만들 예정. step 수 아직

agent 수 알파벳(26)만큼 가능.

python main.py --yaml_path=src_configs/gen_base.yaml

## Streamlit

streamlit run showing_data.py

## Architecture

- agent : agent node 구현, langchain pipeline 생성
- function : function list 저장 및 객체화
- graph : 그래프 관련 모듈 (구현되면 추가 부탁)
- notebook : ipynb
- outputs : 데이터 저장할 곳  
- prompt : 프롬프트들 저장.
- utils : 파일 세이빙 등 유틸
- main.py  
- inference.py (vllm 고쳐야함)

