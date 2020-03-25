class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'fashion_person':
            return "../data/fashion_person/"
        elif dataset == 'fashion_clothes':
            return "../data/fashion_clothes/"
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
