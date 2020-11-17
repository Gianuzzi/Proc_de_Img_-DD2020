# Libs
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Funciones

def Check_IMG(IMG, RGB=False, YIQ=False, normed=False):
    """
    Función para checkear que los valores en una IMG sean válidos
    
    Parámetros:
    ----------
    IMG   : Matriz (ndarray) con los valores de los píxeles a analizar.
    RGB   : Si la matriz de datos es de RGB o Y.
    YIQ   : Si la matriz de datos es de YIQ
    normed: Si la matriz de datos RGB está normalizada a [0,1).
    """

    # Size
    size = IMG.shape
    if (2>len(size)) | (len(size)>3):
        raise ValueError('El array no posee la dimensión adecuada.')
    if len(size)==3:
        if size[2]!=3:
            raise ValueError('El array no posee la dimensión'+\
                             ' en colores adecuada.')

    # RGB
    if RGB:
        tipo   = IMG.ravel().dtype
        floats = ['float32', float]
        ints   = ['uint8', 'uint16', 'uint32', 'uint64',
                  'int16', 'int32', int]
        MAX    = 1 if normed else 255
        if np.any(IMG > MAX):
            raise ValueError('RGB|Y posee valores > %i'%MAX)
        if np.any(IMG < 0):
            raise ValueError('RGB|Y posee valores < 0')
        if (tipo in floats) and not normed:
            raise TypeError('RGB|Y posee valores flotantes')
        if (tipo in ints) and normed:
            raise TypeError('RGB|Y posee solo valores enteros')

    # YIQ
    if YIQ:
        LimI = 0.5957
        LimQ = 0.5226
        if np.any(IMG[:,:,0] < 0    ):
            raise ValueError('IMG posee componente Y < 0')
        if np.any(IMG[:,:,0] > 1    ):
            raise ValueError('IMG posee componente Y > 1')
        if np.any(IMG[:,:,1] < -LimI):
            raise ValueError('IMG posee componente I < {}'.format(-LimI))
        if np.any(IMG[:,:,1] > LimI ):
            raise ValueError('IMG posee componente I > {}'.format(LimI))
        if np.any(IMG[:,:,2] < -LimQ):
            raise ValueError('IMG posee componente Q < {}'.format(-LimQ))
        if np.any(IMG[:,:,2] > LimQ ):
            raise ValueError('IMG posee componente Q > {}'.format(LimQ))
 
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
    Función para modificar el YIQ, por medio de 
     alpha y/o beta.
    
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
    if alpha < 0: raise ValueError('Alpha debe ser > 0')
    if beta  < 0: raise ValueError('Beta debe ser > 0')

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
    Función para aplicar una cierta transformación a un array Y.
    
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
        if   func in ['raíz', 'sqrt']: 
            Y_m = np.sqrt(Y_m)
        elif func in ['cuadrado', 'square']: 
            Y_m = Y_m**2
        elif func in ['constante', 'constant']: 
            Y_m = Y_m*kwargs['c']
        elif func in ['lineal', 'linear']: 
            Y_m = Y_m*kwargs['a'] + kwargs['b']
        elif func in ['potencia', 'power']: 
            Y_m = Y_m**kwargs['p']
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



def Transform_IMG(img, 
                  x_min=0, x_max=1, y_min=0, y_max=1, 
                  x_inv=False, y_inv=False,
                  rot=False, Y=False, verb=False):
    """
    Función para realizar transformaciones a una imagen (ndarray).
    Transformaciones posibles: Crop, Flip, Rotate.
    
    Parámetros:
    ----------
    img   : Imagen (ndarray) a recortar.
    x_min : Posición porcentual [0,x_max), del borde izquierdo
             nuevo, respecto a la imagen original.
    x_max : Posición porcentual (x_min,1], del borde derecho
             nuevo, respecto a la imagen original.
    y_min : Posición porcentual [0,y_max), del borde inferior
             nuevo, respecto a la imagen original.
    y_max : Posición porcentual (y_min,1], del borde superior
             nuevo, respecto a la imagen original.
    x_inv : Inevrtir imagen en x. (Bool)
    y_inv : Inevrtir imagen en y. (Bool)
    rot   : Rotar imagen 90° en sentido antihorario. (Bool)
    verb  : Imrpimir mensaje. (Bool)
    """   
    
    # Check
    if x_min>=x_max: raise ValueError('x_min debe ser < x_max')
    if y_min>=y_max: raise ValueError('y_min debe ser < x_max')
    if (x_min<0) or (x_min>1): 
        raise ValueError('x_min debe ser >= 0 y < 1')
    if (y_min<0) or (y_min>1): 
        raise ValueError('y_min debe ser >= 0 y < 1')
    if (x_max>1): raise ValueError('x_max debe ser <= 1')
    if (y_max>1): raise ValueError('y_max debe ser <= 1')
    
    Check_IMG(img)
    
    # Work on a copy
    img_c = img.copy()
    
    # Crop
    ## Python trabaja desde la esquina superior izquierda,
    ## por eso trabajamos con el opuesto al y otorgado.
    row_min = int((1-y_max) * img.shape[0])
    row_max = int((1-y_min) * img.shape[0]) 
    col_min = int(x_min * img.shape[1])
    col_max = int(x_max * img.shape[1])
    img_c   = img_c[row_min:row_max, col_min:col_max]
    
    # Flip
    if x_inv: img_c = np.flip(img_c, axis=1)
    if y_inv: img_c = np.flip(img_c, axis=0)
    
    # Rotation
    if rot: img_c = np.rot90(img_c)  
        
    # Verbose
    if verb:
        if ((x_max-x_min)!=1) | ((y_max-y_min)!=1):
            print('Se recortó la imagen.')
        if x_inv | y_inv: print('Se invirtió la imagen.')
        if rot: print('Se rotó la imagen.')
    
    return img_c



def Resize_IMG(img,
               width=None, height=None,
               resize=False,
               crop=False,
               fill=None,
               normed=True,
               verb=False):
    """
    Función para realizar edición de tamaño a 1 imagen (array).
    Se define primero el ancho y alto buscado, y luego se 
     especifica qué operaciones se quieren realizar (Ordenadas 
     según orden de ejecución).
        - Cambio de tamaño: La imagen se expande/contrae lo mayor
            posible, conservando su aspecto, hasta el ancho/alto
            especificado.
        - Corte: En caso de que la imagen tenga mayor tamaño al
            especificado, se recorta el excedente.
        - Rellenado: En caso de que la imagen tenga mayor tamaño
            al especificado, se rellena el faltante con algún 
            color (RGB).
    Notesé que, si a no se especifica la operación CROP y/o FILL,
    la imagen puede que no termine con el tamaño especificado.
            
    Parámetros:
    -----------
    img   : Imagen (ndarray) a aplicar edición.
    width : Ancho en píxeles a setear. 
            Si width==None, se mantiene el ancho original. 
            (int)
    height: Alto en píxeles a setear. 
             Si height==None, se mantiene el alto original.
            (int)
    resize: Expandir/Contraer la imagen, conservando el aspecto 
             original (height/width == cte).
            (bool)
    crop  : Aplicar recorte de excedente a la imagen.
            (bool)
    fill  : Aplicar rellenado de faltante, para alcanzar el tamaño
             especificado. 
            Se puede definir una tupla o lista de 3 valores
             (R,G,B), del color a rellenar, o su nombre (en
             inglés), en formato string (Ej.: 'red', 'green'),
             capaz de ser manipulado por matplotlib. 
            Si fill==True se rellena con negro.
            (tuple/list)
    normed: Si la matriz de datos está normalizada a [0,1).
            (bool)
    verb  : Imprimir los pasos realizados.
            (bool)
    """
    
    # Check
    Check_IMG(img, normed=normed)

    # Width/Height
    ## Height
    if isinstance(height, int):
        if   height>0: h = height
        else: raise ValueError('height debe ser > 0')
    elif height is None: h = img.shape[0]
    else: raise TypeError('Formato erróneo de height.')
    ## Width
    if isinstance(width, int): 
        if   width>0: w = width
        else: raise ValueError('width debe ser > 0')
    elif width is None: w = img.shape[1]
    else: raise TypeError('Formato erróneo de width.')
    
    if verb: print('[Ancho, Alto] buscado:', w, h)   
    
    # Working in PIL IMAGE format
    from PIL import Image
    if not normed:
        i = Image.fromarray(img)
    else:
        i = Image.fromarray((img*255).astype('uint8'))
    
    # Verbose
    if verb:
        old = i.size
        print('Tamaño inicial:', old)
        
    # Resize
    if resize:
        if (w<=i.size[0]) | (h<=i.size[1]):
            i.thumbnail([w,h], Image.ANTIALIAS)
        else:
            r = i.size[1]/float(i.size[0])
            sep_w = w / float(i.size[0])
            sep_h = h / float(i.size[1])
            if sep_h < sep_w:
                i = i.resize([int(h/r),h], Image.ANTIALIAS)
            else:
                i = i.resize([w,int(w*r)], Image.ANTIALIAS)
        if verb: 
            print('Aplicado resize:', old, '-->', i.size)
            old = i.size
                
    # Crop | BOX == [left, [up, right), down)
    if crop:
        dw = i.size[0] - w # IMG uses inverted size
        dh = i.size[1] - h # IMG uses inverted size
        le = int(dw/2)     if dw>0 else 0
        ri = int(dw/2) + w if dw>0 else i.size[0]
        up = int(dh/2)     if dh>0 else 0
        do = int(dh/2) + h if dh>0 else i.size[1]
        i  = i.crop((le, up, ri, do))
        if verb: 
            print('Aplicado corte:', old, '-->', i.size)
            old = i.size
            
    # Fill
    if fill is not None:
        wf = w if (i.size[0] < w) else i.size[0]
        hf = h if (i.size[1] < h) else i.size[1]
        if isinstance(fill, list): fill = tuple(fill)
        if fill==True: fill = (0,0,0)
        f = Image.new('RGB', (wf, hf), color=fill)
        f.paste(i, (int((wf - i.size[0]) / 2),
                    int((hf - i.size[1]) / 2)))
        i = f
        if verb: 
            print('Aplicado rellenado:', old, '-->', i.size)
            print('\t Color utilizado:', fill)
    
    # Verbose
    if verb:
        print('\n Tamaño final:', i.size)
        
    # Return to array
    i = np.array(i)
    if normed: i = i/255.
    
    return i



def Resize2IMGs(img1, img2,
                width=None, height=None,
                resize_1=False, resize_2=False,
                crop_1=False, crop_2=False,
                fill_1=None, fill_2=None,
                normed_1=False, normed_2=False,
                verb=False):
    """
    Función para realizar edición de tamaño a 2 imágenes (arrays),
     utilizando la función Resize_IMG. (Ver Resize_IMG? para más
     información.)
     
    Parámetros:
    -----------
    img1    : Imagen 1 (ndarray) a aplicar edición.
    img2    : Imagen 2 (ndarray) a aplicar edición.
    width   : Ancho en píxeles a setear ambas imágenes. 
              Si width==None, ambas conservan el ancho.
              Si width=='first' / 'second', se utiliza el ancho de
              la imagen 1 / 2, para ambas imágenes.
              (int)
    height  : Alto en píxeles a setear ambas imágenes.
              Si height==None, ambas conservan el alto.
              Si height=='first' / 'second', se utiliza el alto de
              la imagen 1 / 2, para ambas imágenes.
              (int)
    resize_1: Aplicar resize en imagen 1.
              (bool)
    resize_2: Aplicar resize en imagen 2.
              (bool)
    crop_1  : Aplicar crop en imagen 1.
              (bool)
    crop_2  : Aplicar crop en imagen 2.
              (bool)
    fill_1  : Aplicar rellenado en imagen 1.
              (bool)
    fill_2  : Aplicar rellenado en imagen 2.
              (bool)
    verb    : Imprimir los pasos realizados.
              (bool)
    """
    
    # Check
    Check_IMG(img1)
    Check_IMG(img2)
    
    # Width/Height
    ## Height
    if   height == 'first' : height = img1.shape[0]
    elif height == 'second': height = img2.shape[0] 
    ## Width
    if   width == 'first' : width = img1.shape[1]
    elif width == 'second': width = img2.shape[1]  
    
    #Transform
    if verb: print('Transformando Imagen 1...')
    i1 = Resize_IMG(img1,
                    width=width, height=height,
                    resize=resize_1,
                    crop=crop_1,
                    fill=fill_1,
                    verb=verb)
    if verb: print('\n Transformando Imagen 2...')
    i2 = Resize_IMG(img2,
                    width=width, height=height,
                    resize=resize_2,
                    crop=crop_2,
                    fill=fill_2,
                    verb=verb)
    
    return i1, i2



def Algebra_IMGs(img1, img2, 
                 normed_1=True, normed_2=True,
                 op=None, fo=None, 
                 verb=False):
    """
    Función para realizar operaciones algebraicas entre 2 
     imágenes (ndarrays). Ambos array deben tener el mismo
     tamaño.
    Las operaciones (píxel a píxel) disponibles son:
        - suma    ['suma',      'sum',       '+']
        - resta   ['resta',     'subtract',  '-']
        - lighter ['mas_claro', 'if_lighter']
        - darker  ['mas_oscuro','if_darker' ]
    Los formatos (cierres) disponible son:
        - RGB clampeado ['RGB_truncado', 'RGB_clamp'  ]
        - RGB promedio  ['RGB_promedio', 'RGB_average']
        - YIQ clampeado ['YIQ_truncado', 'YIQ_clamp'  ] 
        - YIQ promedio  ['YIQ_promedio', 'YIQ_average ]
        - Ninguno       [None] (OBLIGATORIO si se opera
                                 con lighter/darker)
    
    Nota: Al operar en el espacio YIQ, en TODOS los casos
           se termina la operación clampeando las 3 
           componentes en sus respectivos límites.
           
    Algoritmos (en cada píxel):
        |   op    |   fo   |
        ---------------------
        - "suma"  | "RGB_clamp":
            R, G, B := (R1+R2, G1+G2, B1+B2)
        - "resta" | "RGB_clamp":
            R, G, B := (R1-R2, G1-G2, B1-B2)
        - "suma"  | "RGB_promedio":
            R, G, B := (R1+R2, G1+G2, B1+B2)/2.
        - "resta" | "RGB_promedio":
            R, G, B := (R1-R2, G1-G2, B1-B2)/2.
        - "suma"  | "YIQ_clamp":
            Y := (Y1 + Y2)
            I := (Y1 * I1 - Y2 * I2) / (Y1 + Y2)
            Q := (Y1 * Q1 - Y2 * Q2) / (Y1 + Y2)
        - "resta" | "YIQ_clamp":
            Y := (Y1 - Y2)
            I := (Y1 * I1 - Y2 * I2) / (Y1 - Y2)
            Q := (Y1 * Q1 - Y2 * Q2) / (Y1 - Y2)
        - "suma"  | "YIQ_promedio":
            Y := (Y1 + Y2)/2.
            I := (Y1 * I1 - Y2 * I2) / (Y1 + Y2)
            Q := (Y1 * Q1 - Y2 * Q2) / (Y1 + Y2)
        - "resta" | "YIQ_promedio":
            Y := (Y1 - Y2)/2.
            I := (Y1 * I1 - Y2 * I2) / (Y1 - Y2)
            Q := (Y1 * Q1 - Y2 * Q2) / (Y1 - Y2)
        - "mas_claro"
            if Y1 > Y2:
                Y, I, Q := Y1, I1, Q1
            else:
                Y, I, Q := Y2, I2, Q2
        - "mas_oscuro"
            if Y1 < Y2:
                Y, I, Q := Y1, I1, Q1
            else:
                Y, I, Q := Y2, I2, Q2
    
    Parámetros:
    -----------
    img1    : Imagen 1 (ndarray) a aplicar operación.
    img2    : Imagen 2 (ndarray) a aplicar operación.
    normed_1: Si img1 ya está en RGB normalizado.
              (bool)
    normed_2: Si img2 ya está en RGB normalizado.
              (bool)
    op      : Operación a realizar, entre la lista de
               posibles (ver arriba).
              (str)
    fo      : Formato de operación (cierre) a realizar,
               entre la lista de posibles (ver arriba).
              (str)
    verb    : Imprimir el proceso realizado.
              (bool)
    """
    
    # Check
    if img1.shape!=img2.shape:
        raise ValueError('ERROR. Las imágenes deben'+\
                         ' tener el mismo tamaño.')
    ops = ['suma',      'sum',        '+',
           'resta',     'subtract',   '-',
           'mas_claro', 'if_lighter', 
           'mas_oscuro','if_darker']
    
    fos = ['RGB_truncado', 'RGB_clamp',
           'RGB_promedio', 'RGB_average',
           'YIQ_truncado', 'YIQ_clamp',
           'YIQ_promedio', 'YIQ_average']
    
    if op is None: 
        return print('No se especificó operación.')
    if op not in ops: 
        raise ValueError('Operación mal definida.')
        
    if (op in ops[:6]) and (fo not in fos):
        raise ValueError('Formato mal (o no) definido.')
    if (op in ops[6:]) and (fo is not None):
        raise ValueError('Operaciones "if" deben'+\
                         ' tener formato None.')
      
    Check_IMG(img1, RGB=True, normed=normed_1)
    Check_IMG(img2, RGB=True, normed=normed_2)
    
    if not normed_1: img1 = img1/255.
    if not normed_2: img2 = img2/255.
    
    # RGB
    if fo in fos[:4]:
        if   op in ops[ :3]: img = img1 + img2 # sum
        elif op in ops[3:6]: img = img1 - img2 # sub
        if fo in fos[:2]: ## clamp
            img[img>1] = 1
            img[img<0] = 0
        else:             ## average
            if op in ops[3:6]: img += 1 # if sub
            img /= 2.
    
    #YIQ
    else: # includes lighter/darker
        img1 = RGBtoYIQ(img1, normed=True)
        img2 = RGBtoYIQ(img2, normed=True)
        Y1   = img1[:,:,0]
        I1   = img1[:,:,1]
        Q1   = img1[:,:,2]
        Y2   = img2[:,:,0]
        I2   = img2[:,:,1]
        Q2   = img2[:,:,2]
        if op in ops[6:]: # lighter/darker
            if op in ops[6:8]: # lighter
                img = np.maximum(img1, img2)
            else:              # darker
                img = np.minimum(img1, img2)
            msk = np.equal(img [:,:,0], Y1)
            img[:,:,1] = I1 * msk + I2 * ~msk
            img[:,:,2] = Q1 * msk + Q2 * ~msk
        else:
            if   op in ops[:3]: # sum
                img        = img1 + img2
                img[:,:,1] = (Y1 * I1 + Y2 * I2)/(Y1 + Y2)
                img[:,:,2] = (Y1 * Q1 + Y2 * Q2)/(Y1 + Y2)
            elif op in ops[3:6]: # sub
                img        = img1 - img2
                img[:,:,1] = (Y1 * I1 - Y2 * I2)/(Y1 - Y2)
                img[:,:,2] = (Y1 * Q1 - Y2 * Q2)/(Y1 - Y2)
            if fo in fos[6:]:
                if op in ops[3:6]: img[:,:,0] += 1 # if sub
                img[:,:,0] /= 2. ## average
            img[:,:,0][img[:,:,0] < 0    ] = 0          #Y
            img[:,:,0][img[:,:,0] > 1    ] = 1          #Y
            img[:,:,1][img[:,:,1] < -0.5957] = -0.5957  #I
            img[:,:,1][img[:,:,1] > 0.5957 ] = 0.5957   #I
            img[:,:,2][img[:,:,2] < -0.5226] = -0.5226  #Q
            img[:,:,2][img[:,:,2] > 0.5226 ] = 0.5226   #Q
            
        img = YIQtoRGB(img, normed=True, verb=verb)
    
    # Final re-check
    Check_IMG(img, RGB=True, normed=True)
    
    # Verb
    if verb: print('Se realizó la operación:', op,'\n\t'+\
                   'utilizando el ciere:', fo)
    
    return img



def Combinatoria(n, k):
    """
    Función para realizar la operación de combinatoria:
    (n | k)
    """
    up = np.math.factorial(n)
    do = np.math.factorial(k)*np.math.factorial(n-k)
    
    return up/do



def Filter(name, n=3, norm=True, verb=True, **kwargs):
    """
    Función para obtener la matriz de algún filtro.
    
    Parámetros:
    -----------
    
    name  : Nombre del filtro.  (str)
            Los posibles son:
             - 'identidad', 'identity'
             - 'plano', 'plain' 
             - 'gauss', 'gaussiano', 'gaussian' 
             - 'bartlett'
             - 'laplace', 'laplaciano', 'laplacian'
             - 'laplace_bordes', 'laplaciano_bordes', 'laplacian_edges'
             - 'sobel_norte', 'sobel_north'
             - 'sobel_sur', 'sobel_south' 
             - 'sobel_este', 'sobel_east' 
             - 'sobel_oeste', 'sobel_west'
             - 'sobel_noroeste', 'sobel_northwest', 'sobel_no', 'sobel_nw'
             - 'sobel_noreste', 'sobel_northeast', 'sobel_ne'
             - 'sobel_suroeste', 'sobel_southwest', 'sobel_so', 'sobel_sw'
             - 'sobel_sureste', 'sobel_southeast', 'sobel_se'
             - 'linea_horizontal', 'linea_h'
             - 'linea_vertical', 'linea_v'
             - 'linea_45'
             - 'linea_135' 
             - 'combinado', 'combined' [Requiere kwargs]
    n     : Ancho de matriz de filtro. (int)
    norm  : Normalizar el filtro, si la suma de sus valores es > 1. (bool)
    verb  : Imprimir mensaje de filtro creado.
    
    
    kwargs: En caso de seleccionar operación: 'combinado' o 'combined', 
             se ejecuta la función FilterCombine:
                 filtro = filter_1 * c_1 'op' filter_2 * c_2
            Se debe introducir entonces:
                filtro_1: Nombre del filtro 1.
                filtro_2: Nombre del filtro 2.
                n_1     : Ancho de matriz de filtro 1.
                n_2     : Ancho de matriz de filtro 2.
                c_1     : Factor multiplicativo de matriz 1.
                c_2     : Factor multiplicativo de matriz 2.
                op      : Operación a realizar:
                            - 'suma',           'sum',      '+',
                            - 'resta',          'subtract', '-'
                            - 'multiplicacion', 'product',  '*'
                            - 'division',       'divide',   '/'
                            
            En caso de seleccionar operación: 'propio' o 'custom',
             se debe introducir como argumento:
                 matrix: Array con el filtro hecho manualmente.
    """
    
    # Check
    names = ['identidad', 'identity',
             'plano', 'plain', 
             'gauss', 'gaussiano', 'gaussian', 
             'bartlett',
             'laplace', 'laplaciano', 'laplacian',
             'laplace_bordes', 'laplaciano_bordes',
                 'laplacian_edges',
             'sobel_norte', 'sobel_north',
             'sobel_sur', 'sobel_south', 
             'sobel_este', 'sobel_east', 
             'sobel_oeste', 'sobel_west',
             'sobel_noroeste', 'sobel_northwest',
                 'sobel_no', 'sobel_nw',
             'sobel_noreste', 'sobel_northeast',
                 'sobel_ne',
             'sobel_suroeste', 'sobel_southwest',
                 'sobel_so', 'sobel_sw',
             'sobel_sureste', 'sobel_southeast',
                 'sobel_se',
             'linea_horizontal', 'linea_h',
             'linea_vertical', 'linea_v',
             'linea_45',
             'linea_135', 
             'combinado', 'combined',
             'propio', 'custom']
    
    if ((n<1) or isinstance(n, float) or (n%2==0)):
        raise ValueError('n debe ser un entero'+\
                         ' positivo impar > 1.')  
    if name not in names:
        raise ValueError('Nombre erróneo.')    
    
    # Definimos Filtro
    ## Identidad
    if   name in names[:2]: 
        filtro = np.zeros((n,n))
        filtro[n//2, n//2] = 1
    ## Plano
    elif name in names[2:4]:
        filtro = np.ones((n,n))
    ## Gauss
    elif name in names[4:7]: 
        aux = []
        for i in range (n//2 + 1):
            aux.append(Combinatoria(n-1,i))
        aux += aux[-2::-1]
        aux = np.array(aux, dtype=int)
        filtro = np.matmul(aux.T[:, np.newaxis],
                           aux[np.newaxis,:])
    ## Bartlett
    elif name==names[7]:
        aux = []
        for i in range (1, n//2 + 2):
            aux.append(i)
        aux += aux[-2::-1]
        aux = np.array(aux, dtype=int)
        filtro = np.matmul(aux.T[:, np.newaxis],
                           aux[np.newaxis,:])
    ## Laplace (esquinas -1)
    elif name in names[8:11]:
        filtro = np.ones((n,n)) * -1
        filtro[n//2, n//2] = n*n - 1
    ## Laplace (esquinas 0)
    elif name in names[11:14]:
        filtro = np.zeros((n,n))
        filtro[n//2, :] = filtro[:, n//2] = np.ones(n) * -1
        filtro[n//2, n//2] = 2*n - 2
    ## Direccionales NESO (Sobel)
    elif name in names[14:22]:
        aux  = np.arange(-(n//2),n//2+1)[:, np.newaxis]
        aux  = np.repeat(aux, n, axis=1)
        aux0 = aux.copy()
        aux[:,n//2] *=2
        if name in names[14:16]: # N
            filtro = np.flip(aux, axis=0) 
        if name in names[16:18]: # S
            filtro = aux
        if name in names[18:20]: # E
            filtro = aux.T
        if name in names[20:22]: # O
            filtro = np.flip(aux.T, axis=1) 
    ## Direccionales DIAG (Sobel)
    elif name in names[22:36]: 
        aux = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                aux[i,j] = i - j
        if name in names[22:26]: # NO
            filtro = np.flip(aux.T, axis=1)
        if name in names[26:29]: # NE
            filtro = aux.T
        if name in names[29:33]: # SO
            filtro = aux
        if name in names[33:36]: # SE
            filtro = np.flip(aux, axis=1)
    # Líneas
    elif name in names[36:42]:
        if n!=3: print('Solo hay disponibles kernels de'+\
                      ' líneas de tamaño 3x3.')
        aux = np.ones((3,3))*-1
        if name in names[36:38]: # Horiz
            aux[1] = 2
            filtro = aux
        if name in names[38:40]: # Verti
            aux[:,1] = 2
            filtro   = aux
        if name == 'linea_45':   # 45 grados
            np.fill_diagonal(aux, 2)
            filtro = np.flip(aux, axis=1)
        if name == 'linea_135':  # 135 grados
            np.fill_diagonal(aux, 2)
            filtro = aux
    # Combinado (Para posible pasa banda)
    elif name in names[42:44]: 
        filtro = FilterCombine(**kwargs, normed=False)
    # Propio
    elif name in names[44:]:
        filtro = kwargs['matrix']
        
    # Normalize
    if (norm) & (filtro.sum()>0):
        filtro = filtro / filtro.sum()
    
    # Verbose
    if verb & (name not in names[42:]):
        print('Se generó un filtro "{}" {}x{}'.format(
                name, n,n))
        
    return filtro



def FilterCombine(filter_1='gauss', filter_2='gauss',
                  n_1=5, n_2=3,
                  c_1=1, c_2=1,
                  op='*',
                  normed=False, 
                  verb=True):
    """
    Función para realizar operación entre 2 filtros.
    De esta forma se podría obtener un filtro PasaBanda.
    La operación es:
        filtro = filter_1 * c_1 'op' filter_2 * c_2
            donde 'OP' puede ser:
            - 'suma',           'sum',      '+',
            - 'resta',          'subtract', '-'
            - 'multiplicacion', 'product',  '*'
            - 'division',       'divide',   '/'
            
    Se normaliza la matriz al finalizar, si la suma es > 0.
    
    Parámetros:
    ----------
    filter_1: Nombre del filtro 1 a utilizar. Debe ser 
               entendido por la función Filter. (str)
    filter_2: Nombre del filtro 2 a utilizar. Debe ser 
               entendido por la función Filter. (str)
    n_1     : Ancho de matriz de filtro 1. (int)
    n_2     : Ancho de matriz de filtro 2. (int)
    c_1     : Constante multiplicativa de la matriz
               de filtro 1. (int)
    c_2     : constante multiplicativa de la matriz
               de filtro 2. (int)
    op      : Operación algebráica a realizar.
               Las posibles opciones son:
                 - 'suma',           'sum',      '+'
                 - 'resta',          'subtract', '-'
                 - 'multiplicacion', 'product',  '*'
                 - 'division',       'divide',   '/'
               (str)
    normed  : Normalizar el filtro, si la suma de sus
               valores es > 1. (bool)
    verb     : Imprimir mensaje de filtro creado.
    """
    
    # Check
    ops = ['suma',           'sum',      '+',
           'resta',          'subtract', '-',
           'multiplicacion', 'product',  '*',
           'division',       'divide',   '/']

    if op not in ops:
        raise ValueError('Operación errónea.')

    # Get matrices
    f1 = Filter(name=filter_1, n=n_1, norm=False)
    f2 = Filter(name=filter_2, n=n_2, norm=False)
    
    # Set same size
    if n_1>n_2:
        f2 = np.pad(f2, (n_1-n_2)//2, constant_values=0)
    else:
        f1 = np.pad(f1, (n_2-n_1)//2, constant_values=0)
        
    # Operation
    if   op in ops[:3]:
        filtro = f1*c_1   +  f2*c_2
        if verb: ope = '+'
    elif op in ops[3:6]: 
        filtro = f1*c_1   -  f2*c_2
        if verb: ope = '-'
    elif op in ops[6:9]: 
        filtro = (f1*c_1) * (f2*c_2)
        if verb: ope = '*'
    else:                
        filtro = (f1*c_1) / (f2*c_2)
        if verb: ope = '/'
    
    # Normalize
    if (normed) & (filtro.sum()>0):
        filtro = filtro / filtro.sum()
    
    # Verbose
    if verb: 
        print('Se realizó la operación:')
        print('\t filtro = ({} * {}) {} ({} * {})'.format(
                filter_1, c_1, ope, filter_2, c_2))
        
    return filtro



def Chunks(img, shape, verb=False):
    """
    Función para generar chunks de una Imagen, 
     de un tamaño específico.
    
    Parámetros:
    -----------
    img  : Array 2D a generar chuncks.
    shape: Tupla/Lista/Arary 2D con el shape de
            los chunks a generar.
    verb : Imprimir mensaje.
    """
    
    # Check
    ## IMG
    if len(img.shape)!=2: 
        raise ValueError('img debe estar en 2D')
    ## SHAPE
    shape = tuple(shape)
    if len(shape)!=2: 
        raise ValueError('shape debe tener len==2')    
        
    # Tamaño (imagen, chunks)    
    ch_shape = tuple(np.subtract(
                    img.shape, shape) + 1) + shape
    # Tupla de bytes para recorrer
    strides  = img.strides + img.strides
    # Recorremos -> Generamos chunks
    chunks   = np.lib.stride_tricks.as_strided(
                        img, ch_shape, strides)    
    
    # Verbose
    if verb: print('Se han creado chunks de la imagen'+\
                   ' de tamaño:\n {} x {}'.format(
                  chunks.shape[0], chunks.shape[1]))
    
    return chunks



def EnlargeImgY(Y, pad=0):
    """
    Función para crear una imagen que sa la extensión de bordes
     de otra imagen. Debe ser un array 2D.
    
    Parámetros:
    -----------
    Y  : Array 2D con la imagen a agrandar. 
    pad: Cantidad de píxeles a agregar a cada lado.
          En caso de uso para filtro, suele ser: (len(filtro) - 1) // 2"""
    
    # Create new array
    Y_m  = np.pad(Y, pad, constant_values=0)
    hn   = Y_m.shape[0]
    wn   = Y_m.shape[1]

    # New edges
    Y_m[:pad        , pad:(wn-pad)] = np.repeat(Y[0 , :][np.newaxis,:], pad, axis=0)
    Y_m[pad:(hn-pad), (wn-pad):   ] = np.repeat(Y[: ,-1][:,np.newaxis], pad, axis=1)
    Y_m[(hn-pad):   , pad:(wn-pad)] = np.repeat(Y[-1, :][np.newaxis,:], pad, axis=0)
    Y_m[pad:(hn-pad), :pad        ] = np.repeat(Y[: , 0][:,np.newaxis], pad, axis=1)

    # New corners
    Y_m[:pad        , :pad     ] = Y[0 , 0]
    Y_m[:pad        , (wn-pad):] = Y[0 ,-1]
    Y_m[(hn-pad):   , (wn-pad):] = Y[-1,-1]
    Y_m[(hn-pad):   , :pad     ] = Y[-1, 0]
    
    return Y_m



def RGBAtoRGB(RGBA, BgC='white', 
              normed=True, verb=False):
    """
    Función para pasar de RGBA a RGB.
    
    Parámetros:
    ----------
    RGBA  : Matriz (ndarray) con los valores RGBA de los
             píxeles a transformar.
             RGBA.shape == [Alto, Ancho, [R,G,B,A]]
    BgC   : Color de fondo de la imagen. Puede ser un string con el
             nombre de un color (en inglés) interpretable por
             matplotlib.colors (Ej. "white", "red", ...), o un 
             array/tupla/lsita de dimensión 3 con las 3 componentes
             RGB del fondo. 
            En caso de componentes RGB, NO DEBEN ESTAR NORMALIZADAS,
             ya que el normalizado se aplica en el código.
    normed: Si la matriz de datos RGBA ya está normalizada a [0,1).
    verb  : Imprimir mensaje al realizar la transformación.
    """
    
    # Check
    ## Size
    size = RGBA.shape
    if (2>len(size)) | (len(size)>3):
        raise ValueError('El array no posee la dimensión adecuada.')
    if (len(size)==3) & (size[2]==3):
        raise ValueError('La imagen ya está en RGB')
    ## Background
    if isinstance(BgC, str): BgC = colors.to_rgb(BgC)
    elif isinstance(BgC, (tuple, np.ndarray, list)):
        BgC = np.array(BgC)/255.
        if np.shape(BgC)[0]!=3: 
            raise ValueError('BgC debe tener dimensión 3. [RGB]')
    else: raise TypeError('Formato de BgC erróneo.')
    # Transform
    if not normed: RGBA = RGBA / 255.
    RGB = np.zeros((RGBA.shape[0], RGBA.shape[1], 3))
    RGB[:,:,0] = ((1 - RGBA[:,:,3]) * BgC[0]) + (RGBA[:,:,3] * RGBA[:,:,0])
    RGB[:,:,1] = ((1 - RGBA[:,:,3]) * BgC[1]) + (RGBA[:,:,3] * RGBA[:,:,1])
    RGB[:,:,2] = ((1 - RGBA[:,:,3]) * BgC[2]) + (RGBA[:,:,3] * RGBA[:,:,2])
            
    # Verbose
    if verb: print('Se ha transformado de RGBA a RGB, con Fondo ='+\
                   ' {}'.format(BgC))
        
    return RGB



def MorfOp(Y, n=0, op=None, verb=False):
    """
    Función para aplicar una operación morfológica
    a una imagen 2D normalizada en luminancias.
    Las operaciones disponibles son:
        - 'identidad'
        - 'erosion'
        - 'dilatacion'
        - 'apertura'
        - 'cierre'
        - 'borde_exterior', 'borde_ext'
        - 'borde_interior', 'borde_int'
        - 'mediana', median'
        - 'top_hat'
        - 'bottom_hat'
        - 'gradiente'
        - 'media', 'mean'
    
    Parámetros:
    -----------
    Y   : Array 2D con la imagen a la cual se aplicará
           la operación. Debe estar normalizada al 
           rango [0,1). (ndarray 2D)
    n   : Ancho de la matriz del kernel a utilizar. 
           (int)
    op  : Nombre de la operación a realizar, entre
           las posibles. (str)
    verb: Imprimir mensaje al realizar la operación.
           (bool)
    """
    
    # Check
    ops   = ['identidad',
             'erosion',
             'dilatacion',
             'apertura',
             'cierre',
             'borde_exterior', 'borde_ext',
             'borde_interior', 'borde_int',
             'mediana', 'median',
             'top_hat',
             'bottom_hat',
             'gradiente',
             'media', 'mean']
    
    if op not in ops:
        raise ValueError('Operación errónea.')
    if ((n<1) or isinstance(n, float) or (n%2==0)):
        raise ValueError('n debe ser un entero'+\
                         ' positivo impar > 1.') 
        
    # Enlarge Y (to avoid losign edges)
    Y_e = EnlargeImgY(Y, pad=((n - 1) // 2))
    
    # Chunks
    chunks = Chunks(Y_e, [n,n])
    
    # Operate
    if   op == 'identidad':  # Y
        Y_m = Y
    elif op == 'erosion':    # min(Y)
        Y_m = np.min(chunks, axis=(2,3))
    elif op == 'dilatacion': # max(Y)
        Y_m = np.max(chunks, axis=(2,3))
    elif op == 'apertura':   # max(min(Y))
        Y_aux = np.min(chunks, axis=(2,3))
        c_aux = Chunks(Y_aux, [n,n])
        Y_m   = np.max(c_aux, axis=(2,3))
    elif op == 'cierre':     # min(max(Y))
        Y_aux = np.max(chunks, axis=(2,3))
        c_aux = Chunks(Y_aux, [n,n])
        Y_m   = np.min(c_aux, axis=(2,3))
    elif op in ops[5:7]:     # max(Y) - Y
        Y_aux = np.max(chunks, axis=(2,3))
        Y_m   = Y_aux - Y  # Y - min(Y)
    elif op in ops[7:9]:
        Y_aux = np.min(chunks, axis=(2,3))
        Y_m   = Y - Y_aux
    elif op in ops[9:11]:    # median(Y)
        Y_m = np.median(chunks, axis=(2,3))
    elif op == 'top_hat':    # Y - max(min(Y))
        Y_aux = MorfOp(Y_e, n=n, op='apertura')
        Y_m   = Y - Y_aux
    elif op == 'bottom_hat':    # Y - min(max(Y))
        Y_aux = MorfOp(Y_e, n=n, op='cierre')
        Y_m   = Y_aux - Y
    elif op == 'gradiente':  # max(Y) - min(Y)
        Y_aux_1 = np.min(chunks, axis=(2,3))
        Y_aux_2 = np.max(chunks, axis=(2,3))
        Y_m     = Y_aux_2 - Y_aux_1
    elif op in ops[-2:]:     # mean(Y)
        Y_m = np.mean(chunks, axis=(2,3))
        
    # Clamp
    Y_m[Y_m < 0] = 0
    Y_m[Y_m > 1] = 1
    
    # Verbose
    if verb: print('Se ha aplicado la operación'+\
                   ' {}, con un kernel {}x{}.'.format(
                    op, n, n))
    
    return Y_m



def ApplyMorfOp(Y):
    """
    Función para ejecutar MorfOp() de forma
    recursiva e interactiva.
    
    Parámetros:
    -----------
    Y : Array 2D con imagen en luminancias
         normalizado. (ndarray)
    """
    
    # For COLAB bug
    import time
    
    # Y1 and Y2
    Y1  = Y.copy()
    Y2  = Y.copy()
    # Yes/No
    yn  = 'y'
    yns = ['y', 'yes', 's', 'si', 'n', 'no',]
    # Tries
    tries = 0
    trn   = 0
    
    while True:
        # Setting n
        if trn>0:
            print('    {} error{}'.format(
                trn, 'es' if trn>1 else ''))
            if trn>=3:
                print(    'Saliendo.')
                return Y2
        n = input('Introduzca el ancho del kernel:')
        try:
            n = int(n)
        except Exception as e:
            trn += 1
            print(e)
            continue
        if ((n<1) or (n%2==0)):
            trn += 1
            print('n debe ser un entero'+\
                         ' positivo impar > 1.')
            continue
        else:
            break
            
    while True:
        # Operate
        op = input('Introduzca la operación a realizar:')
        op = op.replace(" ", '_').replace("'", '').lower()
        if op in ['identidad', 'original']: # back to original
            print('Retornando a imagen original.')
            Y1 = Y.copy()
            op = 'identidad'
        try:
            Y2    = MorfOp(Y1, n=n, op=op, verb=True)
            tries = 0
        except Exception as e:
            tries += 1
            print(e)
            print('    {} error{}'.format(
                tries, 'es' if tries>1 else ''))
            if tries>=3:
                print(    'Saliendo.')
                return Y2
            continue

        # Plot
        fig, axs = plt.subplots(1,3, sharey=True,
                                dpi=150, figsize=(12,4))
        fig.suptitle('Operación morfológica en luminancias')
        fig.tight_layout()
        axs[0].set_title('Imagen original')
        axs[0].imshow(Y, plt.cm.gray)
        axs[0].axis('off')  
        axs[1].set_title('Imagen utilizada para modificación')
        axs[1].imshow(Y1, plt.cm.gray)
        axs[1].axis('off')
        axs[2].set_title('Imagen modificada por "{}"'.format(op))
        axs[2].imshow(Y2, plt.cm.gray)
        axs[2].axis('off')    
        plt.show()
        
        time.sleep(2) # time for showing image (COLAB bug)

        # Y1 for reusing
        Y1 = Y2.copy()

        # Again?
        cont = 0
        while True:
            yn = input('¿Desea continuar? [Y/N]')
            yn = yn.lower()
            if   yn not in yns:
                cont += 1
                print('    No comprendo.')
                if cont >=3:
                    print('    3 errores ocurridos.')
                    print(    'Saliendo.')
                    return Y2
            elif yn in yns[:4]: break
            else: 
                print('Muchas gracias.')
                return Y2

    return # Unused
            
            
            
def Manual_Resize(Y, new_shape='x1', method='nearest', 
                  sp=False, verb=True):
    """
    Función para cambiar el tamaño de una imagen 
    en luminancias.
    
    Parámetros:
    -----------
    Y         : Imagen 2-D con luminancias normalizadas. 
                (ndarray)
    new_shape : Tamaño nuevo de la imagen buscada. 
                 Puede ser una tupla/lista/array con 2
                 elementos (Alto, Ancho) correspondientes
                 al tamaño nuevo, o puede ser un escalar
                 correspondiente al factor de multiplicación
                 de tamaño de la imagen original. También
                 puede ser un string de con formato 'x{factor}'
                 (anteponiendo una x al factor). [Default='x1']
                 Ej: new_shape = [10, 7] # Tamaño nuevo: (10, 7)
                 Ej: new_shape = 1.5     # Tamaño nuevo: tamaño viejo * 1.5 
                 Ej: new_shape = 'x0.4'  # Tamaño nuevo: tamaño viejo * 0.4
    method    : Método a utlizar.
                 Los disponibles son:
                 - 'nn', 'nearest', 'cercanos', 'vc'          # Vecinos cercanos
                 - 'linear', 'bilinear', 'lineal', 'bilineal' # Bilineal
                 - 'cubic', 'bicubic', 'cubico', 'bicubico'   # Bicúbico
                [Default='nearest']
                (Si la imagen se agranda, los pixeles representando 
                los bordes se obtienen por NN)
    sp        : Utilizar el paquete de scipy.interpolate para
                 realizar las interpolaciones. Caja negra,
                 pero puede ser más rápido. [Default=False]
                 (En caso de realizaar una interpolación
                 bicúbica, sp debe ser True, ya que la 
                 implementación manual aún no está terminada.)
    verb      : Imprimir mensajes. [Default=True]
    """
    

    # Check
    Check_IMG(Y, RGB=True, normed=True)
    h, w = Y.shape[0], Y.shape[1]
    
    ## New_shape
    if isinstance(new_shape, (str, int, float)):
        if isinstance(new_shape, str):
            if new_shape[0] in ['x', 'X']:
                N = np.float(new_shape[1:])
            else:
                raise ValueError('Error en string de '\
                                 'new_shape')
        else:
            N = np.float(new_shape)
        if N <= 0: 
            raise ValueError('new shape debe ser factor '\
                             'positivo')
        hn = int(h * N)
        wn = int(w * N)            

    elif isinstance(new_shape, (list, tuple)):
        ns = np.array(new_shape).astype(int)
        if ns.shape != 2: 
            raise ValueError('new_shape debe tener shape==2')
        if (ns[0] < 2) | (ns[1] < 2):
            raise ValueError('new_shape debe ser > 1 en x e y')
        hn, wn = ns[0], ns[1]
        
    ## Methods
    methods = ['nn', 'nearest', 'cercanos', 'vc',
               'linear', 'bilinear', 'lineal', 'bilineal', 
               'cubic', 'bicubic', 'cubico', 'bicubico']
    method  = method.lower()
    
    if method not in methods: 
        raise ValueError('Error en método')
    
    # Start
    if verb:
        print('Tamaño original de la imagen: '
              ' ({}, {})'.format(h,w))
        print('Tamaño final de la imagen   : '
              ' ({}, {})'.format(hn,wn))
        print('Método de interpolación utilizado: '\
              ' {}'.format(method))
        if sp: print('Se utiliza el paquete de SciPy.')
    
    # Grid Original
    dy_or = 1/float(h)
    dx_or = 1/float(w)
    y_or  = np.arange(0, 1, dy_or) + dy_or * 0.5
    x_or  = np.arange(0, 1, dx_or) + dx_or * 0.5

    # Grid New
    dy_ne = 1/float(hn)
    dx_ne = 1/float(wn)
    y_ne  = np.arange(0, 1, dy_ne) + dy_ne * 0.5
    x_ne  = np.arange(0, 1, dx_ne) + dx_ne * 0.5
    
    if method in methods[:8]: # NN OR BILINEAR
        
        ## NN SciPy
        if (method in methods[:4]) & sp:
            from scipy.interpolate import RegularGridInterpolator
            RGI   = RegularGridInterpolator((y_or, x_or), Y, 
                                            method='nearest', 
                                            bounds_error=False, 
                                            fill_value=None)
            grid  = np.array(np.meshgrid(y_ne, x_ne))
            pts   = grid.T.reshape(-1, 2)
            Y_new = RGI(pts).reshape(hn,wn)
            
            # Clamp
            Y_new[Y_new < 0] = 0
            Y_new[Y_new > 1] = 1
            
            # RETURN NN 
            return Y_new
        
        ## Linear SciPy
        if (method in methods[4:8]) & sp:
            from scipy.interpolate import interp2d
            I2D    = interp2d(x_or, y_or, Y, kind='linear')
            Y_new  = I2D(x_ne, y_ne)
            
            # Clamp
            Y_new[Y_new < 0] = 0
            Y_new[Y_new > 1] = 1
            
            # RETURN BILINEAR
            return Y_new
        
        # IF NOT SCIPY
        
        # Work in X
        binx = np.searchsorted(x_or, x_ne, side='right') # right node N°
        o1x       = binx == 0                            # left problems
        binx[o1x] = 1
        o2x       = binx == w                            # right problems
        binx[o2x] = w - 1
        x2   = x_or[binx]                                # right node pos
        x1   = x2 - dx_or                                # left node pos
        dx2  = x2 - x_ne                                 # distance to x2
        dx1  = x_ne - x1                                 # distance to x1
        binx[np.abs(dx1) < np.abs(dx2)] -= 1             # when NN is left

        # Work in Y
        biny = np.searchsorted(y_or, y_ne, side='right') # upper node N°
        o1y       = biny == 0                            # upper problems
        biny[o1y] = 1
        o2y       = biny == h                            # lower problems
        biny[o2y] = h - 1
        y2   = y_or[biny]                                # lower node pos
        y1   = y2 - dy_or                                # upper node pos
        dy2  = y2 - y_ne                                 # distance to y2
        dy1  = y_ne - y1                                 # distance to y1
        biny[np.abs(dy1) < np.abs(dy2)] -= 1             # when NN is upper

        # Create NEW MESHGRID
        Bx, By = np.meshgrid(binx, biny)

        # Create NN Image
        Y_new  = Y[tuple([By,Bx])]

        # RETURN IF NN
        if method in methods[:4]: return Y_new
        
        # If not NN, then BILINEAR INTERPOLATION
       
        binx[np.abs(dx1) < np.abs(dx2)] += 1    # back to right node
        biny[np.abs(dy1) < np.abs(dy2)] += 1    # back to lower node 

        gx = (~o1x) & (~o2x) # points not outside x edges
        gy = (~o1y) & (~o2y) # points not outside y edges

        # bins x not in edges
        binxg  = binx[gx]
        dx2dx1 = np.array([dx2[gx], dx1[gx]]).T

        # bins y not in edges
        binyg  = biny[gy] 
        dy2dy1 = np.array([dy2[gy], dy1[gy]])

        # 4 points and matrix
        x1, y1 = np.meshgrid(binxg-1, binyg-1)
        x2, y1 = np.meshgrid(binxg,   binyg-1)
        x1, y2 = np.meshgrid(binxg-1, binyg)
        x2, y2 = np.meshgrid(binxg,   binyg)
        matrix = np.array([
                    [Y[tuple([y1,x1])], Y[tuple([y2,x1])]],
                    [Y[tuple([y1,x2])], Y[tuple([y2,x2])]]
                    ])
        
        # Offsets
        of1x = o1x.sum()
        of2x = o2x.sum()
        of1y = o1y.sum()
        of2y = o2y.sum()
        
        # Calculate matrix products
        Y_new = np.einsum('ijk,ki->jk', np.einsum('ij,kijm->kjm',
                                dy2dy1, matrix), dx2dx1)
        
        # Normalization
        Y_new[of1y:(hn-of2y),of1x:(wn-of2x)] /= (dx_or * dy_or)
        
        # Clamp
        Y_new[Y_new < 0] = 0
        Y_new[Y_new > 1] = 1
        
        # RETURN BILINEAR
        return Y_new
    
    else: # BICUBIC
        
        if sp: # Use scipy implementation
            from scipy.interpolate import interp2d
            I2D    = interp2d(x_or, y_or, Y, kind='cubic')
            Y_new  = I2D(x_ne, y_ne)
            
            # Clamp
            Y_new[Y_new < 0] = 0
            Y_new[Y_new > 1] = 1
            
            # RETURN BICUBIC
            return Y_new
        
        else:
            raise Exception('Perdón, aún no está implementada '\
                           'la interpolación bicúbica manual; '\
                            'solo se encuentra la de SciPy '\
                            'disponible por el momento. :(')
            
    return # Unused



def ApplyManual_Resize(img, new_shape='x1',method='nearest',
                       sp=False, verb=True):
    """
    Función para ejecutar Manual_Resize() a una imagen.
    
    Parámetros:
    -----------
    img       : Imagen con valores RGB normalizados.
    new_shape : Tamaño nuevo de la imagen buscada.
                 [Default='x1']
    method    : Método a utlizar.
                 [Default='nearest']
    sp        : Utilizar el paquete de scipy. 
                 [Default=False]
    verb      : Imprimir mensajes. 
                 [Default=True]
    """
    
    # Check
    Check_IMG(img, normed=True)
    
    if len(img.shape)==3: # RGB
        R = Manual_Resize(img[:,:, 0], new_shape, method,
                         sp, verb=verb)
        G = Manual_Resize(img[:,:, 1], new_shape, method,
                         sp, verb=False)
        B = Manual_Resize(img[:,:, 2], new_shape, method,
                         sp, verb=False)
        img_new = np.zeros((R.shape[0], R.shape[1], 3))
        img_new[:,:,0] = R
        img_new[:,:,1] = G
        img_new[:,:,2] = B
        
    elif len(img.shape)==2: # Y
        img_new = Resize_Manual(img[:,:, 0], new_shape, method,
                         sp, verb=verb)
    
    else: raise ValueError('Error en imagen.')
        
    return img_new