# Uso di Google Colab e Jupyter notebook
_Google Colab_ è una piattaforma che permette di eseguire codice direttamente sul cloud (macchina virtuale temporanea dotata di acceleratore GPU) 
 o in locale su server privato. Occorre avere un account Google con accesso a Google Drive dove salvare i propri notebook. 
 Di seguito i passi per programmare in Python (o CUDA C) sul server tramite i notebook gestiti in Colab, sfruttando la GPU disponibile.
 
- Collegare l'applicazione __Colaboratory__ dal _Google Workspace Marketplace_ (`Nuovo/altra applicazione/Colaboratory` solo una volta)
- Aprire un nuovo notebook Colab dal proprio Drive (`Nuovo/altro/Google Colaboratory`) mediante browser _Chrome_, _Firefox_ o _Safari_ 
- Dal menu `Runtime` scegliere `Cambia tipo di runtime` e dal pulsante `connetti` scegliere `connetti a runtime ospitato` e scegliere GPU 
nel menu relativo al tipo di acceleratore HW
- Seguire le istruzioni contenute in [CUDA_lab1.ipynb](https://github.com/giulianogrossi/GPUcomputing/blob/master/lab1/CUDA_lab1.ipynb) per programmare in CUDA C
