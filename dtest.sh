curl -s http://127.0.0.1:11435/api/decide \
  -H "Content-Type: application/json" \
  -d '{"model": "laya:multilingual-fp32", "prompt": "test"}'
