def in_mandelbrot(point, iterations):
    """determines whether point is in mandelbrot set"""
    z = 0
    for i in range(iterations):
        z = z**2 + point
        if abs(z)>2:
            return False

    return True

print(in_mandelbrot(complex(0, 0.5), 1000)) 
