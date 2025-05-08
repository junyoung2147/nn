
tensor 사용법
생성:
tensor() 빈 텐서 생성
tensor(std::vector& a) 모양 벡터를 입력받아 텐서 생성 ex. tensor({2,2,2}), 2X2X2 크기의 0으로 채워진 텐서 생성
tensor(Shape shape) Shape 구조체를 입력받아 Shape 모양의 텐서 생성, 0으로 채워짐
tensor(Shape shape, std::shared_ptr array) 모양과 배열 스마트 포인터 형식의 배열을 받아 텐서 생성

사칙연산: 
numpy의 사칙연산처럼 동작
텐서 op 텐서, 텐서 op 실수 가능 (op는 연산자)

논리연산:
각 원소별로 논리연산을 시행, numpy와 같음

transpose() 전치 연산자, 최하위 2차원을 전치시킴
dot(tensor& b) 행렬곱 연산자, 3차원 이상일 경우에는 최하위 2차원을 행렬곱
reshape(Shape shape) 입력받은 모양으로 모양 변경

() 연산자로 인덱스 접근 가능 2X2X2 모양의 텐서라면 t(1,1,1) 형식으로 1,1,1 위치의 값을 가져올 수 있음
t(1,1) 처럼 열을 가져올 수도 있지만 아직 미구
