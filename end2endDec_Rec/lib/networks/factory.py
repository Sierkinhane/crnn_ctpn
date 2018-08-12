from .VGGnet_test import VGGnet_test


def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
           return VGGnet_test()
        elif name.split('_')[1] == 'train':
           return VGGnet_train()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))
