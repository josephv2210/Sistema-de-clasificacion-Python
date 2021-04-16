#importamos la libreria os que nos permite interactuar con archivos
import os

#definimos esta funcion para buscar directorios dentro del dataset 
#esta funcion recibe como parametros una lista para almacenar las rutas de los dir y la ruta
#esta funcion usa la recursividad para buscar en todos los subdirectorios y retorna la lista
def busca_directorios(num, pat):                
    for f in os.listdir(pat): 
        if os.path.isdir(pat+'/'+f):
            num.append(pat+'/'+f)
            busca_directorios(num, pat+'/'+f)        
    return num

#verifica que se este ejecutando el script
if __name__ == '__main__':

    #archivos dentro de la carpeta
    sub_directorios_raiz = os.listdir('./') 

    #accedemos a la carpeta de las imagenes y usamos la funcion definida para buscar subdirectorios
    sub_directorios_imagenes = busca_directorios([], './imagenes')
    print(f"subdirectorios en carpeta de imagenes: {len(sub_directorios_imagenes)} ")

    #buscamos cuantas imagenes tiene cada directorio y almacenamos en un diccionario
    #las keys del diccionario son los subdirectorios y los values es una lista con las imagenes
    imagenes_dict = {}
    temp = []
    cont = 1
    #tambien se renombran las imagenes dentro del dataset, asignando una numeracion empezando desde el cero
    for i in sub_directorios_imagenes:
        for f in os.listdir(i): 
            if os.path.isfile(i+'/'+f):
                if f.split('.')[-1].lower() == 'jpg':
                    os.rename(i+'/'+f, i+'/'+str(cont)+'.jpg')
                if f.split('.')[-1].lower() == 'jpeg':
                    os.rename(i+'/'+f, i+'/'+str(cont)+'.jpeg')
                cont += 1
                temp.append(i+'/'+str(cont))    
        imagenes_dict[i] = temp
        temp = []
        cont = 0
    
    #imprimimos el total de imagenes por cada subdirectorio y el total de imagenes en el dataset
    suma = 0
    for i in imagenes_dict:
        print(f"{i} : {len(imagenes_dict[i])} imagenes")
        suma += len(imagenes_dict[i])
    print(f"TOTAL IMAGENES: {suma}")

