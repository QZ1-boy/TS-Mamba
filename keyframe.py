def find_max_with_radius(arr, radius=3):
    n = len(arr)
    result = []
    last_max_index = -1  # 跟踪最后一个最大值的索引

    for i in range(n):
        # 查找当前范围内的最大值
        if last_max_index == -1 or (i - last_max_index) > radius:
            # 找到当前范围内的最大值
            max_val = max(arr[i:i+radius])
            max_val_index = arr[i:i+radius].index(max_val) + i
            result.append(max_val)
            last_max_index = max_val_index  # 更新最后一个最大值的索引

    return result

# 示例
arr = [1, 3, 9, 15, 6, 8, 3, 17, 5, 10, 20, 7]
result = find_max_with_radius(arr)
print(result)
