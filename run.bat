@echo off
chcp 65001 >nul
REM ============================================================
REM 运行入口：激活 recsys 环境并执行 python main.py
REM 用法：
REM   run.bat                        -> 全流程（eda→clean→split→feature→recall→rank→rerank）
REM   run.bat --steps eda            -> 只跑 EDA
REM   run.bat --steps rank,rerank --resume  -> 精排+重排，复用已落盘 checkpoint
REM 环境要求：已按 docs/ENV_SETUP.md 建好 conda 环境 recsys
REM ============================================================
cd /d "%~dp0"

call conda activate recsys
if errorlevel 1 (
    echo 环境激活失败，请先按 docs\ENV_SETUP.md 创建并验证 recsys 环境
    pause
    exit /b 1
)

python main.py %*
echo.
echo 执行结束（退出码 %errorlevel%）
pause
