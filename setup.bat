@echo off
echo ============================================
echo   CineMatch - Auto File Arranger
echo ============================================
echo.

echo [1/5] Creating folder structure...
mkdir app 2>nul
mkdir app\api 2>nul
mkdir app\api\v1 2>nul
mkdir app\core 2>nul
mkdir app\db 2>nul
mkdir app\models 2>nul
mkdir app\schemas 2>nul
mkdir app\services 2>nul
mkdir scripts 2>nul
mkdir tests 2>nul
mkdir data 2>nul
mkdir .github 2>nul
mkdir .github\workflows 2>nul
mkdir logs 2>nul
echo    Done.

echo [2/5] Creating __init__.py files...
type nul > app\__init__.py
type nul > app\api\__init__.py
type nul > app\api\v1\__init__.py
type nul > app\core\__init__.py
type nul > app\db\__init__.py
type nul > app\models\__init__.py
type nul > app\schemas\__init__.py
type nul > app\services\__init__.py
type nul > scripts\__init__.py
type nul > tests\__init__.py
echo    Done.

echo [3/5] Moving files into correct folders...

REM app/ root
move /Y main.py          app\main.py           >nul 2>&1

REM app/core/
move /Y config.py        app\core\config.py    >nul 2>&1
move /Y cache.py         app\core\cache.py     >nul 2>&1
move /Y logging.py       app\core\logging.py   >nul 2>&1

REM app/db/
move /Y vector_store.py  app\db\vector_store.py >nul 2>&1

REM app/models/
move /Y embedder.py      app\models\embedder.py >nul 2>&1

REM app/schemas/
move /Y movie.py         app\schemas\movie.py  >nul 2>&1

REM app/services/
move /Y recommendation.py   app\services\recommendation.py   >nul 2>&1
move /Y semantic_search.py  app\services\semantic_search.py  >nul 2>&1
move /Y streaming.py        app\services\streaming.py        >nul 2>&1

REM app/api/v1/
move /Y recommend.py     app\api\v1\recommend.py >nul 2>&1
move /Y router.py        app\api\v1\router.py    >nul 2>&1
move /Y search.py        app\api\v1\search.py    >nul 2>&1

REM scripts/
move /Y ingest.py        scripts\ingest.py     >nul 2>&1

REM tests/
move /Y test_recommend.py tests\test_recommend.py >nul 2>&1

REM .github/workflows/
move /Y ci.yml           .github\workflows\ci.yml >nul 2>&1

REM Move datafolder contents to data/ if user named it datafolder
if exist datafolder\tmdb_5000_movies.csv  move /Y datafolder\tmdb_5000_movies.csv  data\ >nul 2>&1
if exist datafolder\tmdb_5000_credits.csv move /Y datafolder\tmdb_5000_credits.csv data\ >nul 2>&1

echo    Done.

echo [4/5] Fixing requirements.txt for Python 3.13...
(
echo # Web Framework
echo fastapi==0.111.0
echo uvicorn[standard]==0.29.0
echo pydantic==2.7.1
echo pydantic-settings==2.2.1
echo.
echo # ML / Embeddings
echo sentence-transformers==2.7.0
echo torch
echo numpy
echo.
echo # Vector Database
echo chromadb==0.5.0
echo.
echo # Data Processing
echo pandas==2.2.2
echo.
echo # Async HTTP
echo httpx==0.27.0
echo.
echo # Logging
echo loguru==0.7.2
echo.
echo # Testing
echo pytest==8.2.0
echo pytest-asyncio==0.23.6
) > requirements.txt
echo    Done.

echo [5/5] Creating .env file...
if not exist .env (
    copy .env.example .env >nul 2>&1
    echo    Created .env from .env.example
) else (
    echo    .env already exists, skipping
)

echo.
echo ============================================
echo   All done! Folder structure is ready.
echo ============================================
echo.
echo Next steps:
echo   1. Add your Kaggle CSVs into the data\ folder
echo   2. Run:  pip install -r requirements.txt
echo   3. Run:  python scripts\ingest.py
echo   4. Run:  uvicorn app.main:app --reload
echo.
pause
