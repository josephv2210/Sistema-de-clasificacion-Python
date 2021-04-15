import os

#entrenamiento no 96
#entrenamiento yes 155
#test no 98
#test yes 155

def busca_directorios(num, pat):                
    for f in os.listdir(pat): 
        if os.path.isdir(pat+'/'+f):
            num.append(pat+'/'+f)
            busca_directorios(num, pat+'/'+f)        
    return num
if __name__ == '__main__':
    #nombre carpeta root
    #buscamos dentro del archivo root la carpeta del proyecto
    for f in os.listdir('../'):
        if f.lower() == 'sistema-de-clasificacion-python':
            directorio_raiz = f

    #archivos dentro de la carpeta
    sub_directorios_raiz = os.listdir('./') 

    #accedemos a la carpeta de las imagenes
    sub_directorios_imagenes = busca_directorios([], './imagenes')
    print(sub_directorios_imagenes)
    print(f"subdirectorios en carpeta de imagenes: {len(sub_directorios_imagenes)} ")

