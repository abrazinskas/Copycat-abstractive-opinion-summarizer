def update_moving_avg(avg_so_far, new_val, n):
    # First time, n = 1
    new_avg = avg_so_far * (n - 1) / float(n) + new_val / float(n)
    return new_avg
