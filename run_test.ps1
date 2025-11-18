# Script to run RagTool test with required environment variables
$env:EMBEDDINGS_OPENAI_API_KEY = $env:AZURE_API_KEY
Write-Host "✅ Environment variable set: EMBEDDINGS_OPENAI_API_KEY" -ForegroundColor Green
python .\tests\test_rag_tool.py
