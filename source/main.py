import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def get_df_missed_unique_values(df):
    if isinstance(df, pd.DataFrame):
        ret = {}
        for col in df.columns:
            n = df[col].isna().sum()
            n1 = df[col].unique()
            ret[col] = {
                    'manquantes': n,
                    'manquantes (%)': 100*n/df.shape[0],
                    'uniques': n1,
                    'unique (nombre)': len(n1),
                    'uniques (%)': 100*len(n1)/df.shape[0]
                    }
            
        return ret
    return {}


def get_df_basic_elements(df, infos_types= ['shape', 'head', 'tail', 'info', 'describe'], actions= ['return', 'print']):
    ret = { 'shape': '', 'head': '', 'tail': '',  'info': '', 'describe': '' }
    if isinstance(df, pd.DataFrame):
        if 'shape' in infos_types:
            df_shape = df.shape
            if 'print' in  actions: 
                print("shape------------------------------------------------------")
                print(df_shape)
            if 'return' in actions:
                ret['shape'] = str(df_shape)
        if 'head' in infos_types:
            df_head = df.head()
            if 'print' in  actions: 
                print("head------------------------------------------------------")
                print(df_head)
            if 'return' in actions:
                ret['head'] = str(df_head)
        if 'tail' in infos_types:
            df_tail = df.tail()
            if 'print' in  actions: 
                print("tail------------------------------------------------------")
                print(df_tail)
            if 'return' in actions:
                ret['tail'] = str(df_tail)
        if 'info' in infos_types:
            df_info = df.info()
            if 'print' in  actions: 
                print("info------------------------------------------------------")
                print(df_info)
            if 'return' in actions:
                ret['info'] = str(df_info)
        if 'describe' in infos_types:
            df_describe = df.describe()
            if 'print' in  actions: 
                print("describe------------------------------------------------------")
                print(df_describe)
            if 'return' in actions:
                ret['describe'] = str(df_describe)
        
        if 'return' in actions:
            return {'data': ret}
    return {'data': ret}              
    
    
import os
from keras.preprocessing.image import load_img, img_to_array

def load_images_to_dataframe1(images_folders = list(), csv_output_file_path=''):
    from PIL import Image
    import numpy as np
    import sys
    import os
    import csv
    if isinstance(images_folders, list):
        if len(images_folders) > 0:
            for dir in images_folders:
                files_in_folder = sorted(os.listdir(dir))
                files_in_folder.pop(0)  # Remove any non-image files if needed
                for img in files_in_folder:
                    img_file = Image.open(dir + img)
                    # img_file.show()

                    # get original image parameters...
                    width, height = img_file.size
                    format = img_file.format
                    mode = img_file.mode

                    # Make image Greyscale
                    img_grey = img_file.convert('L')
                    #img_grey.save('result.png')
                    #img_grey.show()

                    # Save Greyscale values
                    value = np.asarray(img_grey.getdata(), dtype=np.int64).reshape((img_grey.size[1], img_grey.size[0]))
                    value = value.flatten() # flattern to be ready to write to CSV file
                    # print(value)
                    # write to csv file
                    with open(csv_output_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(value)


def load_images_to_dataframe(images_folders = list(), csv_output_file_path='', newdimensions=[32, 32], grayscale= True ):
    """Charger l'ensemble des images à partir des dossiers dans un dataframe 

    Args:
        images_folders (list, obligatoire): Liste des chemins où se trouvent les images.
        csv_output_file_path (str, optional): chemin de fichier CSV à générer pour stocker numériquement les images, plus son ID et le product_id qu'elle repésente. 
        Defaults to '' ce qui génèrera le fichier images.csv dans le dossier courant.
        
    Noms de chaque image : image_id1_product_id2.png
    avec :
    id1 : représente l'id unique de l'images
    id2 : représente l'id du produit qu'elle représente
    
    exemple : 'image_1303778026_product_4159406790.jpg'   
    """
    dict = {'image': None, 'height': None, 'width': None,  'imageid': None,  'productid': None}
    img_df =pd.DataFrame(data=dict, index=None, columns=['image', 'image_size', 'imageid', 'productid'])
    if isinstance(images_folders, list):
        if len(images_folders) > 0:
            for dir in images_folders:
                files_in_folder = sorted(os.listdir(dir))
                files_in_folder.pop(0)  # Remove any non-image files if needed
                # don't limit max cols and rows when writting output to terminal or file
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)

                img_df = pd.DataFrame()
                for img in files_in_folder:
                    # extract id1 and id2 from image name
                    id_image = img[6:14]
                    id_product = img[23:-4]
                    
                    img = os.path.abspath(image_folder + 'image_' + str(id_image) + '_product_' + str(id_product) + '.jpg' )
                    from PIL import Image
                    # Load the original image
                    original_image = Image.open(img)
                    # Resize the image
                    dimension = (newdimensions[0], newdimensions[1])
                    loaded_img = original_image.resize(dimension)
                    if grayscale : loaded_img = loaded_img.convert('L')
                    
                    #loaded_img = load_img(img).convert('L') # 'L' fir grayscale
                    # reduce size to 32x32
                    #loaded_img = loaded_img.resize((int(32),int(32)), loaded_img.ANTIALIAS)
                    img_array = img_to_array(loaded_img).astype(float)
                    
                    
                    # loaded_img = load_img(dir + img).convert('L')
                    # img_array = img_to_array(loaded_img).astype(int)
                    img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], img_array.shape[2]).flatten()
                    dict = {'image': img_array, 'height': img_array.shape[0], 'width': img_array.shape[1], 'imageid': id_image,  'productid': id_product}
                    img_df = pd.concat([img_df, pd.DataFrame.from_dict(dict)], ignore_index=True, axis=0)
            #img_df = pd.DataFrame(str)
            if img_df is not None:        
                # img_df.to_csv('./csv_images/' + img[0:-4] + '.csv')
                np.set_printoptions(threshold=sys.maxsize) # avoid trancating long values (see https://stackoverflow.com/questions/53316471/pandas-dataframes-to-csv-truncates-long-values)
                img_df.to_csv(csv_output_file_path)
            
    
def load_image_To_DataFrame(image_folder, id_image, id_product, newdimensions=[32, 32], grayscale= True ):
    """Charger dans un DataFrame l'image avec 'image_id1_product_id2.png' 
        avec :
    Args:
        image_folder : dossier où se trouve le fichier image
        id1 : représente l'id unique de l'image
        id2 : représente l'id du produit qu'elle représente
        
    Return:
        DataFrame représenting image with colomns:
            'image': np array, 
            'image_size': image size,  
            'imageid': image id,  
            'productid': product id that image represents       
    """
    
    try:
        dict = {'image': None, 'height': None, 'width': None,  'imageid': None,  'productid': None}
        img_df =pd.DataFrame(data=dict, index=None, columns=['image', 'height', 'width', 'imageid', 'productid'])
        img = os.path.abspath(image_folder + 'image_' + str(id_image) + '_product_' + str(id_product) + '.jpg' )
        
        
        from PIL import Image
        # Load the original image
        original_image = Image.open(img)
        # Resize the image
        dimension = (newdimensions[0], newdimensions[1])
        loaded_img = original_image.resize(dimension)
        if grayscale : loaded_img = loaded_img.convert('L')
        
        #loaded_img = load_img(img).convert('L') # 'L' fir grayscale
        # reduce size to 32x32
        #loaded_img = loaded_img.resize((int(32),int(32)), loaded_img.ANTIALIAS)
        img_array = img_to_array(loaded_img).astype(float)
        img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], img_array.shape[2]).flatten()
        dict = {'image': img_array, 'height': loaded_img.height, 'width': loaded_img.width, 'imageid': id_image,  'productid': id_product}
        img_df = pd.concat([img_df, pd.DataFrame.from_dict(dict)], ignore_index=True, axis=0)
        if img_df.shape[0]>=1:
            return img_df
        else:
            return None
    except:
        print("An exception occurred in function load_image_To_DataFrame !")
        return None


def get_words_from_image(image_path):
    import easyocr as ocr
    from IPython.display import Image
    from pylab import rcParams
    
    reader = ocr.Reader(['en', 'fr'])
    output_reader = reader.readtext(image_path)
    return output_reader
    

    
def my_main():
    """
    df_y = pd.read_csv('Y_train_CVw08PX.csv', sep=',', index_col=0)
    ret = get_df_basic_elements(df_y, ['shape', 'head', 'tail', 'info', 'describe'], ['return', 'print'])
    print('ret=', ret)
    """
    # test :
    # load_images_to_dataframe(images_folders= ['./images/test_images/'], csv_output_file_path='./images.csv')
    
    #load_images_to_dataframe(images_folders= ['./images/image_test/'], csv_output_file_path='./images/image_test/images.csv', newdimensions=[32, 32], grayscale=True)
    #load_images_to_dataframe1(images_folders= ['./images/image_train/'], csv_output_file_path='./images.csv')
    
    
    ret = get_words_from_image(image_path='./images/test_images/image_99638230_product_1080207.jpg')
    
    return 0
    
    # load image as dataframe
    img_df = load_image_To_DataFrame('./images/test_images/', 995104014, 303051753, newdimensions=[32, 32], grayscale=True)
    print(img_df)
    print(img_df.image)
    
    # show image using Matplotlib
    plt.subplot(1, 1, 1) 
    img = np.array(img_df.image).reshape(img_df.iloc[0:1, 1][0], img_df.iloc[0:1, 2][0])
    _ = plt.imshow(img, cmap = 'gray') 
    _ = plt.axis("off") 
    _ = plt.title("Image with id=" + '995104014' + ' and productid=' + '303051753')
    
    
    
    return 0


if __name__ == "__main__":
    sys.exit(my_main())
