functions {
    matrix createAngleMatrix(int complexity, vector angle_vector) {
        int N = num_elements(angle_vector);
        matrix[N, complexity] angle_matrix;
        for (i in 1:complexity) {
            angle_matrix[, i] = angle_vector .^ (i - 1);
        }
        return angle_matrix;
    }
}
    
data {
        int<lower=0> N;   // number of data items
        int<lower=0> P;   // number of beta parameters
        matrix[N, P] Y;    // polynomial features for sam to transform; or a column of ones, angles as floats and true counts for calibration
        vector[N] y;      // ground truth for sam output or input estimates for calibration
        vector[5] angles; // 5 unique angles in the data
    }
    // this step does some transformations to the data
    transformed data{
        // https://mc-stan.org/docs/stan-users-guide/multiple-indexes-with-vectors-and-matrices.html    stan indices start from 1! we add 1 to python indices
        int complexity = 4; // how many parameters for the variance estimation. A quadratic function of the angle has 3 parameters.
        vector[N] angle_vector = Y[, 2];  
        matrix[N, complexity] angle_matrix = createAngleMatrix(complexity, angle_vector);
}

    parameters {
        vector[complexity] beta;      // coefficients 
        vector[complexity] variance_parameters; // variance parameters. not variance itself!
    }
    // Any varianceiable declared as a transformed parameter is part of the output produced for draws.
    transformed parameters{
        vector<lower = 0>[N] variance_train = angle_matrix * variance_parameters;
    }
    model {    
        // in stan, y ~ normal(mu, sigma); y ~ multi_normal(mu, Sigma); Sigma = sigma ^ 2
        beta ~ normal(0, 10);
        variance_parameters ~ normal(0, 10);
        // if we dont put constraints in transformed parameters then we can use exp here
        //vector[N] variance_train = exp(angle_matrix * variance_parameters);
        y ~ normal(angle_matrix[, :complexity] * beta + Y[, 3], variance_train ^ 0.5);
        //y ~ normal(Y[, 3], variance_train ^ 0.5); // for the ai calibration is unnecessary
    }
    generated quantities {
        vector[5] bias = createAngleMatrix(complexity, angles) * beta;
        vector[5] variance = (createAngleMatrix(complexity, angles) * variance_parameters);
    }
