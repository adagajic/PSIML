# imports for reading images
from PIL import Image
import numpy as np
from collections import deque
import heapq

teleports = []
def get_shortest_path(matrix, same_group, border_pixels, teleport = False):
    visited = set()
    # Initialize the priority queue with the source pixels
    # Each item in the queue is a tuple (priority, pixel)
    # The priority is the path length so far
    queue = []
    for pixel in same_group:
        heapq.heappush(queue, (0, pixel))
        visited.add(pixel)
        if teleport and pixel in teleports:
            for t in teleports:
                if pixel != t:
                    visited.add(t)
                    heapq.heappush(queue, (1, t))
            teleport = False
    while queue:
        # Dequeue a pixel from the queue
        path_length, pixel = heapq.heappop(queue)
        if pixel in border_pixels and pixel not in same_group:
            return path_length
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pixel = (pixel[0] + dx, pixel[1] + dy)
            if (0 <= new_pixel[0] < len(matrix) and 0 <= new_pixel[1] < len(matrix[0]) and
                    new_pixel not in visited and matrix[new_pixel[0]][new_pixel[1]] == 0):
                visited.add(new_pixel)
                heapq.heappush(queue, (path_length + 1, new_pixel))
                if teleport and new_pixel in teleports:
                    for t in teleports:
                        if new_pixel != t:
                            visited.add(t)
                            heapq.heappush(queue, (path_length + 2, t))
                            break
                    teleport = False
    return np.inf
    
    
if __name__ == '__main__':
    # Load the images from .png files
    path = input()
    num_teleports = int(input())
    for i in range(num_teleports):
        teleports.append(tuple(map(int, input().split())))
    img = Image.open(path)
    img = np.array(img)
    # count all unique colors in the image
    x = img.shape[0]
    y = img.shape[1]
    matrix = np.ndarray((x, y), dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            if img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][2] == 255:
                matrix[i][j] = 0
            elif img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 2
    # get to every white pixel on borders
    border_white_pixels = []
    for i in range(x):
        if matrix[i][0] == 0:
            if (i, 0) not in border_white_pixels:
                border_white_pixels.append((i, 0))
        if matrix[i][y-1] == 0:
            if (i, y-1) not in border_white_pixels:    
                border_white_pixels.append((i, y-1))
    for i in range(y):
        if matrix[0][i] == 0:
            if (0, i) not in border_white_pixels:
                border_white_pixels.append((0, i))
        if matrix[x-1][i] == 0:
            if (x-1, i) not in border_white_pixels:
                border_white_pixels.append((x-1, i))
    min_distance = np.inf
    count_entr = 0
    min_distance_teleport = np.inf
    for pixel in border_white_pixels:
        if (pixel[0] == 0 or pixel[0] == x-1) and (
            pixel[1] - 1 > 0 and matrix[pixel[0]][pixel[1] - 1] == 0):
            continue
        elif (pixel[1] == 0 or pixel[1] == y-1) and (
            pixel[0] - 1 > 0 and matrix[pixel[0] - 1][pixel[1]] == 0):
            continue
        count_entr += 1
        same_group = set()
        pixel_iter = pixel
        if pixel_iter[0] == 0 or pixel_iter[0] == x-1:
            while pixel_iter[1] <  y and matrix[pixel_iter[0]][pixel_iter[1]] == 0:
                same_group.add(pixel_iter)
                pixel_iter = (pixel_iter[0], pixel_iter[1] + 1)
            pixel_iter = pixel
            while pixel_iter[1] >= 0 and matrix[pixel_iter[0]][pixel_iter[1]] == 0:
                same_group.add(pixel_iter)
                pixel_iter = (pixel_iter[0], pixel_iter[1] - 1)
        elif pixel_iter[1] == 0 or pixel_iter[1] == y-1:
            while pixel_iter[0] < x and matrix[pixel_iter[0]][pixel_iter[1]] == 0:
                same_group.add(pixel_iter)
                pixel_iter = (pixel_iter[0] + 1, pixel_iter[1])
            pixel_iter = pixel
            while pixel_iter[0] >= 0 and matrix[pixel_iter[0]][pixel_iter[1]] == 0:
                same_group.add(pixel_iter)
                pixel_iter = (pixel_iter[0] - 1, pixel_iter[1])
        # get the shortest path to the border pixel
        min_distance = min(min_distance, get_shortest_path(matrix, same_group, border_white_pixels))
        min_distance_teleport = min(min_distance_teleport, get_shortest_path(matrix, same_group, border_white_pixels, True))
    print(count_entr)
    if min_distance == np.inf:
        print(-1)
    else:
        print(min_distance + 1)
    if min_distance_teleport == np.inf:
        print(-1)
    else:
        print(min_distance_teleport + 1)
    