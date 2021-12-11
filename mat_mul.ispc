export void multiply(uniform int a[], uniform int b[], uniform int r[], uniform int width, 
                     uniform int height, uniform int mid) {
    foreach (i = 0...height) {
        for (uniform int j = 0; j < width; j++) {
            r[j][i] = 0;
            for (uniform int k = 0; k < mid; k++) {
                r[j][i] += a[j][k] * b[k][i];
            }
        }
    }
}