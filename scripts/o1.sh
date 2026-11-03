curl -sS -w '\nHTTP %{http_code}\n' \
  http://127.0.0.1:11435/api/decide \
  -H 'Content-Type: application/json' \
  -d '{"model":"laya:multilingual","preset":"triage","state":"I was charged twice for my subscription this month. Please refund the second charge."}'
