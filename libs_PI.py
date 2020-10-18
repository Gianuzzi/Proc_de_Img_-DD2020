import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set()

def Check_IMG(IMG, RGB=False, YIQ=False, normed=False):
    """
    Función para checkear que los valores en una IMG sean válidos
    Parámetros:
    ----------
    IMG   : Matriz (ndarray) con los valores de los píxeles a analizar.
    RGB   : Si la matriz de datos es de RGB.
    YIQ   : Si la matriz de datos es de YIQ.
    normed: Si la matriz de datos RGB está normalizada a [0,1).
    """
    
    # Size
    size = IMG.shape
    if (len(size)!=3) or (size[2]!=3): 
        raise ValueError('El array no posee la dimensión adecuada: [Alto, Ancho, 3]')
       
    # RGB
    if RGB:
        tipo   = IMG.ravel().dtype
        floats = ['float32', float]
        ints   = ['uint8', 'uint16', 'uint32', 'uint64', 'int16', 'int32', int]
        MAX    = 1 if normed else 255
        if np.any(IMG > MAX)              : raise ValueError('RGB posee valores > %i'%MAX)
        if np.any(IMG < 0)                : raise ValueError('RGB posee valores < 0')
        if (tipo in floats) and not normed: raise TypeError('RGB posee valores flotantes')
        if (tipo in ints) and normed      : raise TypeError('RGB posee solo valores enteros')
    
    # YIQ
    if YIQ:
        LimI = 0.5957
        LimQ = 0.5226
        if np.any(IMG[:,:,0] < 0    ): raise ValueError('IMG posee componente Y < 0')
        if np.any(IMG[:,:,0] > 1    ): raise ValueError('IMG posee componente Y > 1')
        if np.any(IMG[:,:,1] < -LimI): raise ValueError('IMG posee componente I < {}'.format(-LimI))
        if np.any(IMG[:,:,1] > LimI ): raise ValueError('IMG posee componente I > {}'.format(LimI))
        if np.any(IMG[:,:,2] < -LimQ): raise ValueError('IMG posee componente Q < {}'.format(-LimQ))
        if np.any(IMG[:,:,2] > LimQ ): raise ValueError('IMG posee componente Q > {}'.format(LimQ))

            
    return 



def RGBtoYIQ(RGB, normed=False, verb=False):
    """
    Función para pasar de RGB a YIQ. 
    Parámetros:
    ----------
    RGB   : Matriz (ndarray) con los valores RGB de los
             píxeles a transformar.
             RGB.shape == [Alto, Ancho, [R,G,B]]
    normed: Si la matriz de datos RGB ya está normalizada a [0,1).
    verb  : Imprimir mensaje al realizar la transformación.
    only_Y: Si se quiere devolver solamente la componente Y.
    """
    
    # Check
    Check_IMG(IMG=RGB, RGB=True, normed=normed)
    
    # Norm
    if not normed: RGB = RGB / 255.    
    
    # Transform
    MAT = np.array([[0.29722594,  0.58780866,  0.1149654 ],
                    [0.59218915, -0.27283   , -0.31935915],
                    [0.21021204, -0.52201776,  0.31180572]])
    
    YIQ  = np.zeros(shape=RGB.shape)     
    for col in range(RGB.shape[0]):
        YIQ[col] = np.matmul(MAT, RGB[col].T).T
    
    
    # Message
    if verb: print('Se ha transformado de RGB a YIQ')
        
    
    return YIQ



def ModifyYIQ(YIQ, alpha=1., beta=1., verb=False):
    """
    Función para modificar el YIQ. 
    Parámetros:
    ----------
    YIQ  : Matriz (ndarray) con los valores YIQ de
              los píxeles a transformar.
              YIQ.shape == [Alto, Ancho, [Y,I,Q]]
    alpha: Escalar positivo a multiplicar la componente Y.
    beta : Escalar positivo a multiplicar las componentes I y Q.
    verb : Imprimir mensaje al realizar las multiplicaciones.
    """
    
    # Check
    Check_IMG(IMG=YIQ, YIQ=True)
    if alpha<0: raise ValueError('Alpha debe ser > 0')
    if beta<0 : raise ValueError('Beta debe ser > 0')
    
    # Change
    YIQ_m         = YIQ.copy()
    YIQ_m[:,:,0] *= alpha #Y
    YIQ_m[:,:,1] *= beta  #I
    YIQ_m[:,:,2] *= beta  #Q       
    
    # Clamping
    LimI = 0.5957
    LimQ = 0.5226
    
    YIQ_m[:,:,0][YIQ_m[:,:,0] > 1    ] = 1      #Y
    YIQ_m[:,:,1][YIQ_m[:,:,1] < -LimI] = -LimI  #I
    YIQ_m[:,:,1][YIQ_m[:,:,1] > LimI ] = LimI   #I
    YIQ_m[:,:,2][YIQ_m[:,:,2] < -LimQ] = -LimQ  #Q
    YIQ_m[:,:,2][YIQ_m[:,:,2] > LimQ ] = LimQ   #Q
    
    
    # Message
    if verb: print('Se ha multiplicado: Y * {} ; I * {} ; Q * {}'\
               .format(alpha, beta, beta))
    
    
    return YIQ_m   



def YIQtoRGB(YIQ, normed=False, verb=False):
    """
    Función para pasar de YIQ a RGB. 
    Parámetros:
    ----------
    YIQ   : Matriz (ndarray) con los valores RGB de
             los píxeles a transformar.
             RGB.shape == [Alto, Ancho, [Y,I,Q]]
    normed: Si la matriz de datos RGB a devolver 
             será normalizada a [0,1).
    verb  : Imprimir mensaje al realizar la transformación.
    """
    
    # Check
    Check_IMG(IMG=YIQ, YIQ=True)
    
    # Transform
    MAT = np.array([[1,  0.9663,  0.6210],
                    [1, -0.2721, -0.6474],
                    [1, -1.1070,  1.7046]])
    
    RGB = np.zeros(shape=YIQ.shape)    
    for col in range(RGB.shape[0]):
        RGB[col] = np.matmul(MAT, YIQ[col].T).T
    
    # Limits
    RGB[RGB > 1] = 1   
    RGB[RGB < 0] = 0
    
    # Norm
    if not normed: RGB = (RGB*255).astype('uint8')
    
    # Message
    if verb: print('Se ha transformado de YIQ a RGB')
    
    
    return RGB



def PiecewiseLinear(Y, Y_min=0, Y_max=1):
    """
    Función para aplicar una función lineal a trozos a un array:
        Y = 0                           para Y < Y_min
        Y = (Y-Y_min) / (Y_max-Y_min)   para Y_min <= Y <=Y_max
        Y = 1                           para Y > Y_max
    Supuesto para array normalizado al rango [0,1).
    Parámetros:
    ----------
    Y    : Array al que se le aplicará la transformación.
    Y_min: Valor de Y, tal que se Y == 0 para todo Y < Y_min.
    Y_min: Valor de Y, tal que se Y == 1 para todo Y > Y_max.
    """
    
    # Check   
    if Y_min < 0    : raise ValueError('Y_min debe ser >= 0')
    if Y_max > 1    : raise ValueError('Y_max debe ser <= 1')
    if Y_max < Y_min: raise ValueError('Y_max debe ser >= Y_min')
    
    # Transform
    Y_m          = Y.copy()    
    l            = Y_m < Y_min
    h            = Y_m > Y_max
    Y_m[l]       = 0
    Y_m[h]       = 1
    Y_m[~l & ~h] = (Y_m[~l & ~h] - Y_min)/ (Y_max - Y_min)
    
    return Y_m



def ChangeY(Y, func=None, clamp=True, **kwargs):
    """
    Función para aplicar una cierta tranformación a un array Y.
    Parámetros:
    ----------
    Y   : Array al que se le aplicará la transformación.
    func: Función a aplicar. Puede ser una función built-in llamable
           (EJ.: np.mean, np.sqrt, etc.; tener en cuenta que se 
           deberá introducir los argumentos necesarios), o el nombre
           de una de la siguiente lista:
            - ['raíz', 'sqrt']          
                -> Para raíz cuadrada: Y = sqrt(Y)
            - ['cuadrado', 'square']    
                -> Para elevar al cuadrado: Y = Y * Y
            - ['constante', 'constant'] 
                -> Para multiplicar por una constante: Y = Y * c
                   Se debe introducir como argumento el parámetro:
                       c = <valor>
            - ['lineal', 'linear']
                -> Para aplicar función lineal: Y = Y * a + b
                   Se deben introducir como argumento los parámetros:
                       a = <valor>
                       b = <valor> 
            - ['potencia', 'power']
                -> Para elevar a cierta potencia: Y = Y**p
                   Se debe introducir como argumento el parámetro:
                       p = <valor> (Potencia)
            - ['lineal_a_trozos', 'PiecewiseLinear']
                -> Para aplicar función a trozos: (Ver PiecewiseLinear?)
                   Se deben introducir como argumento los parámetros:
                       Y_min = <valor>
                       Y_max = <valor>
    clamp: Aplicar clamping para acotar valores al rango [0,1).
    """
    
    # Function names
    funcs = ['raíz', 'sqrt', 
             'cuadrado', 'square',
             'constante', 'constant',
             'lineal', 'linear',
             'potencia', 'power',
             'lineal_a_trozos', 'PiecewiseLinear']  
    
    # Start
    print('Se aplica la función:', func)
    Y_m = Y.copy()
    
    # Callable functions
    if callable(func): Y_m = func(Y_m, **kwargs)  
        
    # Named functions
    elif isinstance(func, str):
        if   func in ['raíz', 'sqrt']         : Y_m = np.sqrt(Y_m)
        elif func in ['cuadrado', 'square']   : Y_m = Y_m**2
        elif func in ['constante', 'constant']: Y_m = Y_m*kwargs['c']
        elif func in ['lineal', 'linear']     : Y_m = Y_m*kwargs['a'] + kwargs['b']
        elif func in ['potencia', 'power']    : Y_m = Y_m**kwargs['p']
        elif func in ['lineal_a_trozos', 'PiecewiseLinear']:
            Y_m = PiecewiseLinear(Y_m, **kwargs)
            
        else:
            raise ValueError('Función desconocida. Las fuciones'+\
                             ' posibles son:\n {}, o una llamable.'\
                             .format(funcs))

    # None
    elif func==None: 
            print('No se aplcian cambios')
            return Y_m
    
    # Else
    else: raise TypeError('Formato de función desconocido.')
    
    if clamp:
        # Limits
        Y_m[Y_m > 1] = 1
        Y_m[Y_m < 0] = 0 
        
        
    return Y_m