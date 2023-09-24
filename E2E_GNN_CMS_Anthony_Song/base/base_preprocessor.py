from abc import abstractmethod

class BasePreprocessor:
    '''
    Base class for all preprocessors
    '''
    def __init__(self,use, data_dir,*args):
        self.use = use
        self.data_dir = data_dir
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if self.use:
            self.preprocess(*args)
        
    @abstractmethod
    def preprocess(*args):
        '''
        preprocess the given raw data
        '''
        raise NotImplementedError

    def get_train_dataset(self):
        if self.train_dataset is None:
            raise NotImplementedError
        return self.train_dataset
    
    def get_val_dataset(self):
        if self.val_dataset is None:
            raise NotImplementedError
        return self.val_dataset
    
    def get_test_dataset(self):
        if self.test_dataset is None:
            raise NotImplementedError
        return self.test_dataset
