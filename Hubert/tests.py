labels = {
            0: 'LUp+',
            1: 'LDown+',
            2: 'LUp-',
            3: 'LDown-',
            4: 'RUp+',
            5: 'RDown+',
            6: 'RUp-',
            7: 'RDown-'
        }

from many_body_builder import create_determinant_from_labels

print(bin(create_determinant_from_labels(labels, ['LUp+', 'RDown+'])))