echo off
cd "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\Debug"
echo "" >> "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\mcs.csv"
for /L %%i in (0,1,0) do (
"C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\Debug\mpso-gpu.exe" config\mcs\config_mcs-%%i.txt >> "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\mcs.csv"
)
python "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\mpso-gpu\utils\cleanup_headers.py" mcs
