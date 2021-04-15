import os

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
    print(f"subdirectorios en carpeta de imagenes: {len(sub_directorios_imagenes)} ")

    #buscamos cuantas imagenes tiene cada directorio y almacenamos en un diccionario
    imagenes_dict = {}
    temp = []
    cont = 1
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
    
    #total de imagenes
    suma = 0
    for i in imagenes_dict:
        print(f"{i} : {len(imagenes_dict[i])} imagenes")
        suma += len(imagenes_dict[i])
    print(f"TOTAL IMAGENES: {suma}")

