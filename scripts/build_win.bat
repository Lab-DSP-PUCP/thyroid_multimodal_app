@echo off
set NAME=ThyroidAid
pyinstaller ^
  --noconfirm ^
  --clean ^
  --name %NAME% ^
  --onefile ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --add-data "models;models" ^
  app.py
