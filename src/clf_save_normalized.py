import classifier


def main():
    X_train, y_train = classifier.load_data()
    ########################################################################       
    # Normalizing                                                               
    ########################################################################       

    X_train_n1 = classifier.normalize(X_train, mode='total')                              
    X_train_n2 = classifier.normalize(X_train, mode='each')                               

    classifier.save_to_file(X_train_n1, y_train, 'normalized_means_by_total.csv', 'normalized')
    classifier.save_to_file(X_train_n1, y_train, 'normalized_means_by_each.csv', 'normalized')
    return


if __name__ == '__main__':
    main()
