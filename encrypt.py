import os
import cv2
import queue
import numpy as np
from scipy.fftpack import dct


huffmanTable = {}

def divideNonOverlapping(carrier_image:np.ndarray):
    height = carrier_image.shape[0]
    width = carrier_image.shape[1]
    stride = 8
    patches = ()
    for x in range(0, height-stride+1, stride):
        for y in range(0, width-stride+1, stride):
            patches += (carrier_image[x:x+stride, y:y+stride], )
    return patches


def splicePatches(patches, row, col):
    image = ()
    for i in range(row):
        for j in range(col):
            bar += patches(i*col+j)

class HuffmanTreeNode(object):
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
    def __lt__(self, other):
        return (self.value < other.value)
    def children(self):
        return ((self.left, self.right))


def buildTree(priotityQueue):
    while True: 
        node1 = priorityQueue.get()
        node2 = priorityQueue.get()
        parent_node = HuffmanTreeNode(node1.value + node2.value, node1, node2)
        if priorityQueue.empty() is True:
            break
        else:
            priorityQueue.put(parent_node)
    return parent_node


def dfs(node:HuffmanTreeNode, word):
    if node.left is None: # Leaf node
        huffmanTable[node.right] = word
    else:
        dfs(node.left, word+b'\0')
        dfs(node.right, word+b'\1')
    return


if __name__=="__main__":
    """
    Step 1: DCT
    Divide the carrier image into non overlapping blocks of size 8x8 and apply DCT on each of the blocks of the cover image f to obtain F using eq^n(1).
    """
    test_img = cv2.imread('fedora.jpg')
    test_patches = divideNonOverlapping(test_img)
    forwardCoefficient = ()
    for patch in test_patches:
        forwardCoefficient += (dct(patch, 1), )
    #print('first F matrix:', forwardCoefficient[0])
    """
    Step 2: Huffman encoding
    Perform Huffman encoding on the 2-D secret image S of size M2xN2 to convert it into a 1-D bits stream H
    """
    secret = cv2.imread('blackcorner.jpg')
    iiu8 = np.iinfo(secret.dtype)
    num_channels = len(secret.shape)
    frequency = {} #np.zeros( (iiu8.max+1,)*num_channels, dtype=np.int )
    for row in secret:
        for nd_pixel in row:
            pixel = tuple(nd_pixel)
            if pixel in frequency:
                frequency[pixel] += 1
            else:
                frequency[pixel] = 1
    print('frequency size:', len(frequency))
    #temp = np.unravel_index(np.argsort(frequency.flatten()), frequency.shape)
    #flattened_f = frequency.flatten()
    priorityQueue = queue.PriorityQueue()
    for pixel, value in frequency.items():#for i in range(flattened_f.size):
        if value > 0: #if flattened_f[i] > 0:
            new_node = HuffmanTreeNode(value, None, pixel)
            #new_node = HuffmanTreeNode(flattened_f[i], None, np.unravel_index(i, frequency.shape))
            print('pixel:{}, frequence:{}'.format(pixel, value))
            priorityQueue.put(new_node)

    root = buildTree(priorityQueue)
    dfs(root, b'')
    #for pixel, word in huffmanTable.items():
    #print('pixel_value:{},word:{}'.format(pixel, word))
    bitstream = b''
    for row in secret:
        for nd_pixel in row:
            pixel = tuple(nd_pixel)
            bitstream = bitstream + huffmanTable[pixel]
    print('length of 1-D bit stream:', len(bitstream))
    """
    Step 3: 8-bit block preparation
    Huffman code H is decomposed into 8-bits blocks B
    """
    print('DCT of first block', forwardCoefficient[0])

    """
    Step 5:
    Perform the inverser block DCT on F using eq^n(2) and obtain a new image f_1 which contains secret image.
    """
    # Problem: idct results are not all integers, thus lose information of secret image when dct again to extract


