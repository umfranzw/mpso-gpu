rmdir "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\Debug\config" /s /q

xcopy "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\mpso-gpu\config" "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\Debug\config" /e /i

rmdir "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\Debug\kernels" /s /q

xcopy "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\mpso-gpu\kernels" "C:\Users\Wayne\Documents\Visual Studio 2010\Projects\mpso-gpu\Debug\kernels" /e /i