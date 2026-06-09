import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title="유튜브 요약 & 번역기", page_icon="📺", layout="centered")

st.title("📺 유튜브 영상 요약 & 번역기")
st.markdown("""
유튜브 영상 링크를 입력하면, 인공지능이 영상의 자막을 분석하여 **한국어로 요약 및 번역**해 줍니다.
*(주의: 자막이 제공되는 영상만 가능합니다)*
""")

# 사이드바 설정 (API 키 입력 및 설정)
# with st.sidebar:
#     st.header("설정")
#     openai_api_key = st.text_input("OpenAI API Key 입력", type="password")
#     model_name = st.selectbox("OpenAI 모델 선택", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
#     target_language = st.selectbox("번역할 언어", ["한국어", "영어", "일본어", "중국어"])
    
#     st.markdown("---")
#     st.markdown("### 필수 설치 라이브러리")
#     st.code("pip install streamlit langchain langchain-openai youtube-transcript-api", language="bash")

    # 2. 사이드바에서 API 키 입력 받기
with st.sidebar:
    st.header("설정")
    # 환경 변수에 OPENAI_API_KEY가 있으면 자동으로 가져옵니다.
    model_name = st.selectbox("OpenAI 모델 선택", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    target_language = st.selectbox("번역할 언어", ["한국어", "영어", "일본어", "중국어"])
    api_key = st.text_input(
        "OpenAI API Key", 
        value=os.environ.get("OPENAI_API_KEY", ""), 
        type="password",
        placeholder="sk-..."
    )
    st.markdown("---")
    st.caption("발급받으신 `sk-`로 시작하는 API 키를 입력해 주세요.")

# 메인 화면 (URL 입력)
youtube_url = st.text_input("유튜브 영상 URL을 입력하세요:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("요약 및 번역 시작", type="primary"):
    if not api_key:
        st.error("왼쪽 사이드바에서 OpenAI API Key를 먼저 입력해주세요.")
    elif not youtube_url:
        st.warning("유튜브 URL을 입력해주세요.")
    else:
        with st.spinner("영상 자막을 추출하고 요약하는 중입니다. 잠시만 기다려주세요..."):
            try:
                # 1. 유튜브 자막 추출
                # add_video_info=False로 설정하여 pytube 의존성 문제 방지 (자막만 확실하게 추출)
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url, 
                    add_video_info=False,
                    # 한국어, 영어 자막을 우선적으로 찾습니다.
                    language=["ko", "en", "en-US"] 
                )
                docs = loader.load()
                
                if not docs:
                    st.error("이 영상에서 자막을 추출할 수 없습니다. (자막이 비활성화되어 있거나 제공되지 않는 영상입니다)")
                else:
                    transcript = docs[0].page_content
                    
                    # 2. LangChain & OpenAI 설정
                    llm = ChatOpenAI(temperature=0.3, model=model_name, api_key=api_key)
                    
                    # 프롬프트 템플릿 생성
                    prompt = PromptTemplate.from_template(
                        """당신은 전문 요약가이자 번역가입니다. 
                        아래 제공된 유튜브 영상의 자막을 분석하여 핵심 내용을 요약하고, 반드시 {language}로 번역하여 출력해주세요.
                        
                        [요청 사항]
                        1. 전체 내용의 핵심 주제를 한 줄로 요약할 것.
                        2. 주요 내용을 3~5개의 불릿 포인트(-)로 나누어 상세히 설명할 것.
                        3. 가독성 좋게 마크다운 형식으로 작성할 것.
                        
                        [영상 자막]
                        {transcript}
                        """
                    )
                    
                    # 3. LangChain 파이프라인(LCEL) 구성 및 실행
                    chain = prompt | llm | StrOutputParser()
                    
                    result = chain.invoke({
                        "language": target_language,
                        "transcript": transcript
                    })
                    
                    # 4. 결과 출력
                    st.success("요약 및 번역이 완료되었습니다!")
                    st.markdown("---")
                    st.markdown(result)
                    
                    # 원본 자막 확인 기능 (옵션)
                    with st.expander("원본 자막 텍스트 보기"):
                        st.text_area("자막 데이터", transcript, height=200)
                        
            except Exception as e:
                st.error(f"오류가 발생했습니다. URL을 확인하거나 잠시 후 다시 시도해주세요.\n에러 내용: {e}")