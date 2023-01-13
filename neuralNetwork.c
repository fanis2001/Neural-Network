#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define d 2 
#define K 3 
#define H1 15
#define H2 15
#define H3 15
#define learning_rate 0.0001
#define TRAIN 4000
#define TEST 4000

typedef double (*ActivationFunction)(double);

typedef struct{
    double** w;
    double* u;
    double* y;
    double* di;
    double** den_w;
    double** de_w;
    ActivationFunction activ;
}Layer;

struct pairs{
    double x1;
    double x2;
};

struct batches{
    struct pairs* Pairs;
};

double sigmoid(double x){
  return 1.0 / (1.0 + exp(-x));
}

double relu(double x){
  return (x > 0) ? x : 0;
}

double tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double linear(double x){
    return x;
}

void loadExamples(struct pairs *s){ 
    FILE *fp = fopen("examplesSDT.txt","r");

    int i = 0;
    double x1;
    double x2;

    while(i!=8000){
        if(fscanf(fp,"%lf %lf",&x1,&x2) != 2){
            perror("Error reading data from file");
        }else{
            s[i].x1 = x1;
            s[i].x2 = x2;
            i++;
        }
    }

    fclose(fp);
}

char* findCategory(double* x){
    
    char* category;

    for(int i = 0;i < 8000;i++){

        if((double)pow((x[0]-0.5),2.0) + (double)pow((x[1]-0.5),2.0)<0.2 && x[1]>0.5){
            category = "C1";
        }else if((double)pow((x[0]-0.5),2.0) + (double)pow((x[1]-0.5),2.0)<0.2 && x[1]<0.5){
            category = "C2";
        }else if((double)pow((x[0]+0.5),2.0) + (double)pow((x[1]+0.5),2.0)<0.2 && x[1]>-0.5){
            category = "C1";
        }else if((double)pow((x[0]+0.5),2.0) + (double)pow((x[1]+0.5),2.0)<0.2 && x[1]<-0.5){
            category = "C2";
        }else if((double)pow((x[0]-0.5),2.0) + (double)pow((x[1]+0.5),2.0)<0.2 && x[1]>-0.5){
            category = "C1";
        }else if((double)pow((x[0]-0.5),2.0) + (double)pow((x[1]+0.5),2.0)<0.2 && x[1]<-0.5){
            category = "C2";
        }else if((double)pow((x[0]+0.5),2.0) + (double)pow((x[1]-0.5),2.0)<0.2 && x[1]>0.5){
            category = "C1";
        }else if((double)pow((x[0]+0.5),2.0) + (double)pow((x[1]-0.5),2.0)<0.2 && x[1]<0.5){
            category = "C2";
        }else{
            category = "C3";
        }
    }
    return category;
}

void print_weights(Layer layer, int neurons_curr_layer, int neurons_prev_layer) {
    for (int i = 0; i < neurons_curr_layer; i++){
        printf("BIAS w[%d][0] = %f\n",i,layer.w[i][0]);
        for (int j = 1; j < neurons_prev_layer+1; j++) {
            printf("w[%d][%d] = %f \n",i,j, layer.w[i][j]);
        }
    }
}

void print_uy(Layer layer, int neurons_curr_layer){
    for(int i = 0; i < neurons_curr_layer;i++){
        printf("u[%d] = %f\n",i,layer.u[i]);
        printf("y[%d] = %f\n\n",i,layer.y[i]);
    }
}

void print_di(Layer layer,int neurons_curr_layer){
    for(int i = 0;i < neurons_curr_layer;i++){
        printf("d[%d] = %f\n",i,layer.di[i]);
    }
}

void print_den_w(Layer layer,int neurons_curr_layer,int neurons_prev_layer){
    for (int i = 0; i < neurons_curr_layer; i++){
        for (int j = 0; j < neurons_prev_layer+1; j++){
            printf("den_w[%d][%d] = %.4f\n",i,j,layer.den_w[i][j]);
        }
    }
}

void print_de_w(Layer layer,int neurons_curr_layer,int neurons_prev_layer){
    for (int i = 0; i < neurons_curr_layer; i++){
        for (int j = 0; j < neurons_prev_layer+1; j++){
            printf("de_w[%d][%d] = %.4f\n",i,j,layer.de_w[i][j]);
        }
    }
}

void init_layer(Layer* layer, int neurons_curr_layer, int neurons_prev_layer, ActivationFunction activ) {
    
    layer->w = malloc(neurons_curr_layer * sizeof(double*));
    for (int i = 0; i < neurons_curr_layer; i++) {
        layer->w[i] = malloc((neurons_prev_layer+1) * sizeof(double));
    }

    layer->den_w = malloc(neurons_curr_layer * sizeof(double*));
    for (int i = 0; i < neurons_curr_layer; i++) {
        layer->den_w[i] = malloc((neurons_prev_layer+1) * sizeof(double));
    }

    layer->de_w = malloc(neurons_curr_layer * sizeof(double*));
    for (int i = 0; i < neurons_curr_layer; i++) {
        layer->de_w[i] = malloc((neurons_prev_layer+1) * sizeof(double));
    }
    for(int i = 0;i < neurons_curr_layer;i++){
        for(int j = 0;j < neurons_prev_layer+1;j++){
            layer->de_w[i][j] = 0;
        }
    }
    layer->u = malloc(neurons_curr_layer * sizeof(double));
    layer->y = malloc(neurons_curr_layer * sizeof(double));
    layer->di = malloc(neurons_curr_layer * sizeof(double));
    layer->activ = activ;

}

void init_input_layer(Layer* layer,double* input){
    layer->y = malloc(d * sizeof(double));
    layer->u = malloc(d * sizeof(double));
    layer->u = input;
    layer->y = input;

}

void init_weights(Layer* h1, Layer* h2, Layer* h3, Layer* out) {
    //srand(time(NULL));
    
    for (int i = 0; i < H1; i++) {
        for (int j = 0; j < d+1; j++) {
            h1->w[i][j] = (double)rand() / RAND_MAX * 2 - 1;
        }
    }
    
    for (int i = 0; i < H2; i++) {
        for (int j = 0; j < H1+1; j++) {
            h2->w[i][j] = (double)rand() / RAND_MAX * 2 - 1;
        }
    }

    for (int i = 0; i < H3; i++) {
        for (int j = 0; j < H2+1; j++) {
            h3->w[i][j] = (double)rand() / RAND_MAX * 2 - 1;
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < H3+1; j++) {
            out->w[i][j] = (double)rand() / RAND_MAX * 2 - 1;
        }
    }
}

void calculate_u_y(Layer *layer,double* input, int neurons_curr_layer,int neurons_prev_layer){
    for(int i = 0;i < neurons_curr_layer;i++){
        layer->u[i] = 0;
        for(int j = 0;j < neurons_prev_layer+1; j++){
            if(j == 0){
                layer->u[i] += layer->w[i][j];
            }else{
                layer->u[i] += layer->w[i][j]*input[j-1];
            }
        }
        layer->y[i] = layer->activ(layer->u[i]);
    }
}

void forward_pass(Layer* input,Layer* h1,Layer* h2,Layer* h3,Layer* out){
    calculate_u_y(h1,input->y,H1,d);
    calculate_u_y(h2,h1->y,H2,H1);
    calculate_u_y(h3,h2->y,H3,H2);
    calculate_u_y(out,h3->y,K,H3);
}

void calculate_output_error_sigmoid(Layer* out,double* t){
    for(int i = 0;i < K;i++){
        out->di[i] = out->y[i]*(1-out->y[i])*(out->y[i]-t[i]);
    }
}

void calculate_hidden_error_relu(Layer* currH, Layer* nextH, int neurons_currH,int neurons_nextH){
    for(int i = 0;i < neurons_currH;i++){
        double sum = 0;
        for(int j = 0;j < neurons_nextH;j++){
            sum += nextH->w[j][i+1]*nextH->di[j];
        }
        if (currH->u[i]>0){
            currH->di[i] = 1*sum;
        }else if(currH->u[i] <= 0){
            currH->di[i] = 0;
        }
    }    
}

void calculate_hidden_error_sigmoid(Layer* currH, Layer* nextH, int neurons_currH,int neurons_nextH){
    for(int i = 0;i < neurons_currH;i++){
        double sum = 0;
        for(int j = 0;j < neurons_nextH;j++){
            sum += nextH->w[j][i+1]*nextH->di[j];
        }
        currH->di[i] = currH->y[i]*(1-currH->y[i])*sum;
    }    
}

void calculate_hidden_error_tanh(Layer* currH, Layer* nextH, int neurons_currH,int neurons_nextH){
    for(int i = 0;i < neurons_currH;i++){
        double sum = 0;
        for(int j = 0;j < neurons_nextH;j++){
            sum += nextH->w[j][i+1]*nextH->di[j];
        }
        currH->di[i] = (1-(currH->y[i]*currH->y[i]))*sum;
    }    
}

void partial_derivative(Layer* currLayer,Layer* prevLayer,int neurons_curr_layer, int neurons_prev_layer){
    for(int i = 0;i < neurons_curr_layer;i++){
        currLayer->den_w[i][0] = currLayer->di[i];
        for(int j = 1;j < neurons_prev_layer+1;j++){
            currLayer->den_w[i][j] = currLayer->di[i]*prevLayer->y[j-1];
        }
    }
}

void backpropagation(Layer* h1, Layer* h2, Layer* h3, Layer* out,Layer* input){
    double t1[3] = {1,0,0};
    double t2[3] = {0,1,0};
    double t3[3] = {0,0,1};

    char* c = findCategory(input->y); 

    if(strcmp(c,"C1") == 0){
        calculate_output_error_sigmoid(out,t1);
        
    }else if(strcmp(c,"C2") == 0){
        calculate_output_error_sigmoid(out,t2);
    }else if(strcmp(c,"C3") == 0){
        calculate_output_error_sigmoid(out,t3);
    }
    
    if(h3->activ == relu){
        calculate_hidden_error_relu(h3,out,H3,K);
    }else if(h3->activ == sigmoid){
        calculate_hidden_error_sigmoid(h3,out,H3,K);
    }else if(h3->activ == tanh){
        calculate_hidden_error_tanh(h3,out,H3,K);
    }

    if(h2->activ == relu){
        calculate_hidden_error_relu(h2,h3,H2,H3);
    }else if(h2->activ == sigmoid){
        calculate_hidden_error_sigmoid(h2,h3,H2,H3);
    }else if(h2->activ == tanh){
        calculate_hidden_error_tanh(h2,h3,H2,H3);
    }
    
    
    if(h1->activ == relu){
        calculate_hidden_error_relu(h1,h2,H1,H2);
    }else if(h1->activ == sigmoid){
        calculate_hidden_error_sigmoid(h1,h2,H1,H2);
    }else if(h1->activ == tanh){
        calculate_hidden_error_tanh(h1,h2,H1,H2);
    }
    
    partial_derivative(out,h3,K,H3);
    partial_derivative(h3,h2,H3,H2);
    partial_derivative(h2,h1,H2,H1);
    partial_derivative(h1,input,H1,d);

}

void update_weights_team(Layer* layer,int neurons_curr_layer,int neurons_prev_layer){
    for(int i = 0;i < neurons_curr_layer;i++){
        for(int j = 0;j < neurons_prev_layer+1;j++){
            layer->w[i][j] = layer->w[i][j] - learning_rate*layer->de_w[i][j];
        }
    }
}

void update_weights_serial(Layer* layer,int neurons_curr_layer,int neurons_prev_layer){
    for(int i = 0;i < neurons_curr_layer;i++){
        for(int j = 0;j < neurons_prev_layer+1;j++){
            layer->w[i][j] = layer->w[i][j] - learning_rate*layer->den_w[i][j];
        }
    }
}

void calculate_squad_error(Layer* out, double *t, double *e){
    *e = 0;
    for(int i = 0;i < K;i++){
        *e += (double)pow(t[i] - out->y[i],2);
    }

    *e = (double)*e/2;
}

void init_de_w(Layer* layer, int neurons_curr_layer,int neurons_prev_layer){
    for(int i = 0;i < neurons_curr_layer;i++){
        for(int j = 0;j < neurons_prev_layer+1;j++){
            layer->de_w[i][j] = 0;
        }
    }
}

void calculate_de_w(Layer* layer,int neurons_curr_layer, int neurons_prev_layer){
    for(int i = 0;i < neurons_curr_layer;i++){
        for(int j = 0;j < neurons_prev_layer+1;j++){
            layer->de_w[i][j] += layer->den_w[i][j];
        }
    }
}

void decisions(Layer* out,char* category,double* correct,double* false,int* decis){
    
    if(strcmp(category,"C1") == 0){
        if(out->y[0] > out->y[1] && out->y[0] > out->y[2]){
            *correct += 1;
            *decis = 1;
        }else{
            *false += 1;
            *decis = -1;
        }
    }else if(strcmp(category,"C2") == 0){
        if(out->y[1] > out->y[0] && out->y[1] > out->y[2]){
            *correct += 1;
            *decis = 1;
        }else{
            *false += 1;
            *decis = -1;
        }
    }else if(strcmp(category,"C3") == 0){
        if(out->y[2] > out->y[0] && out->y[2] > out->y[1]){
            *correct += 1;
            *decis = 1;
        }else{
            *false += 1;
            *decis = -1;
        }
    }
}

void gradient_descent_serial(Layer* input,Layer* h1, Layer* h2,Layer* h3,Layer* out,ActivationFunction activ1,ActivationFunction activ2,ActivationFunction activ3){
    struct pairs allPairs[8000];
    int t = 0;
    double en = 0;
    double t1[3] = {1,0,0};
    double t2[3] = {0,1,0};
    double t3[3] = {0,0,1};

    init_layer(h1,H1,d,sigmoid);
    init_layer(h2,H2,H1,sigmoid);
    init_layer(h3,H3,H2,sigmoid);
    init_layer(out,K,H3,sigmoid);
    init_weights(h1,h2,h3,out);
    loadExamples(allPairs);

    FILE *fp = fopen("squad_error_per_epoch.txt", "w");
    
    while(1){
        printf("--------Epoxi %d--------\n",t);
        double e = 0;
        double e_check[2] = {0,0};

        for(int i = 0;i < TRAIN;i++){
            double x[2] = {allPairs[i].x1,allPairs[i].x2};

            init_input_layer(input,x);

            forward_pass(input,h1,h2,h3,out);
            backpropagation(h1,h2,h3,out,input);

            char* c = findCategory(x);

            if(strcmp(c,"C1") == 0){
                calculate_squad_error(out,t1,&en);
            }else if(strcmp(c,"C2") == 0){
                calculate_squad_error(out,t2,&en);
            }else if(strcmp(c,"C3") == 0){
                calculate_squad_error(out,t3,&en);
            }

            e += en;
            if(i % 2 == 0){
                e_check[0] = e;
            }else if(i % 2 == 1){
                e_check[1] = e;
            }
            update_weights_serial(h1,H1,d);
            update_weights_serial(h2,H2,H1);
            update_weights_serial(h3,H3,H2);
            update_weights_serial(out,K,H3);
        }
        printf("E(w) = %f     %f\n",e,fabs(e_check[0] - e_check[1]));
        fprintf(fp,"%d %f\n",t,e);

        if(t > 700 && fabs(e_check[0] - e_check[1]) < 0.01){
            fclose(fp);
            break;
        }else{
            t++;
        }
    }
}

void gradient_descent_team(Layer* input,Layer* h1, Layer* h2,Layer* h3,Layer* out,ActivationFunction activ1,ActivationFunction activ2,ActivationFunction activ3,int b){
    struct pairs allPairs[8000];
    int t = 0;
    double en = 0;
    double t1[3] = {1,0,0};
    double t2[3] = {0,1,0};
    double t3[3] = {0,0,1};
    double e_check[2] = {0,0};
    int start = 0;
    int end = b;
    int teams = TRAIN/b;
    struct batches mini_batches[teams];

    init_layer(h1,H1,d,activ1);
    init_layer(h2,H2,H1,activ2);
    init_layer(h3,H3,H2,activ3);
    init_layer(out,K,H3,sigmoid);
    init_weights(h1,h2,h3,out);
    loadExamples(allPairs);
    
    for(int i = 0; i < teams; i++) {
        mini_batches[i].Pairs = malloc(b * sizeof(struct pairs));
        for(int j = 0; j < b; j++) {
            mini_batches[i].Pairs[j].x1 = allPairs[start + j].x1;
            mini_batches[i].Pairs[j].x2 = allPairs[start + j].x2;            
        }
        start += b;
    }
    if(start >= TRAIN){
        start = 0;
    }
    
    FILE *fp = fopen("squad_error_per_epoch.txt", "w");
    
    while(1){
        init_de_w(h1,H1,d);
        init_de_w(h2,H2,H1);
        init_de_w(h3,H3,H2);
        init_de_w(out,K,H3);
        double e = 0;

        
        printf("--------Epoxi %d--------\n",t);

        for(int i = 0;i < teams;i++){
            for(int j = 0;j < b;j++){
                double x[2] = {mini_batches[i].Pairs[j].x1,mini_batches[i].Pairs[j].x2};
                init_input_layer(input,x);
                forward_pass(input,h1,h2,h3,out);
                backpropagation(h1,h2,h3,out,input);
                calculate_de_w(out,K,H3);
                calculate_de_w(h3,H3,H2);
                calculate_de_w(h2,H2,H1);
                calculate_de_w(h1,H1,H2);
                

                char* c = findCategory(x);

                if(strcmp(c,"C1") == 0){
                    calculate_squad_error(out,t1,&en);
                }else if(strcmp(c,"C2") == 0){
                    calculate_squad_error(out,t2,&en);
                }else if(strcmp(c,"C3") == 0){
                    calculate_squad_error(out,t3,&en);
                }
                e += en;
            }
        }
        update_weights_team(h1,H1,d);
        update_weights_team(h2,H2,H1);
        update_weights_team(h3,H3,H2);
        update_weights_team(out,K,H3);

        if(start >= TRAIN){
            start = 0;
        }

        if(t % 2 == 0){
            e_check[0] = e;
        }else if(t % 2 == 1){
            e_check[1] = e;
        }
        printf("E(w) = %f     %f\n",e,fabs(e_check[0] - e_check[1]));
        fprintf(fp,"%d %f\n",t,e);
        if(t > 700 && fabs(e_check[0] - e_check[1]) < 0.01){
            fclose(fp);
            break;
        }else{
            t++;
        }
    }
}

void test(Layer* input,Layer* h1, Layer* h2,Layer* h3,Layer* out,double* correct, double* false){
    struct pairs allPairs[8000];
    loadExamples(allPairs);
    int decis = 0;

    FILE *fp = fopen("test.txt", "w");
    
    for(int i = TRAIN;i < 8000;i++){
        double x[2] = {allPairs[i].x1, allPairs[i].x2};
        init_input_layer(input,x);
        char* c = findCategory(x);
        forward_pass(input,h1,h2,h3,out);
        decisions(out,c,correct,false,&decis);
        if(decis == 1){
            fprintf(fp,"%f %f %s +\n",allPairs[i].x1,allPairs[i].x2,c);
        }else if(decis == -1){
            fprintf(fp,"%f %f %s -\n",allPairs[i].x1,allPairs[i].x2,c);
        }
        decis = 0;
    }
    fclose(fp);
}

int main(){
    int B = 0;
    Layer input,h1,h2,h3,out;
    int act1,act2,act3;
    double correct = 0,false = 0;
    ActivationFunction act_h1,act_h2,act_h3;
    printf("SELECT ACTIVATION FUNCTION FOR H1:\n");
    printf("1)Sigmoid\n");
    printf("2)Relu\n");
    printf("3)Tanh\n");
    printf("Select 1-3: \n");
    if(scanf("%d",&act1) != 1){
        perror("Error reading data from standard input");
    }

    if(act1 == 1){
        act_h1 = sigmoid;
    }else if(act1 == 2){
        act_h1 = relu;
    }else if(act1 == 3){
        act_h1 = tanh;
    }
    printf("---------------------------------------------\n");
    printf("SELECT ACTIVATION FUNCTION FOR H2:\n");
    printf("1)Sigmoid\n");
    printf("2)Relu\n");
    printf("3)Tanh\n");
    printf("Select 1-3: \n");
    if(scanf("%d",&act2) != 1){
        perror("Error reading data from standard input");
    }

    if(act2 == 1){
        act_h2 = sigmoid;
    }else if(act2 == 2){
        act_h2 = relu;
    }else if(act2 == 3){
        act_h2 = tanh;
    }
    printf("---------------------------------------------\n");
    printf("SELECT ACTIVATION FUNCTION FOR H3:\n");
    printf("1)Sigmoid\n");
    printf("2)Relu\n");
    printf("3)Tanh\n");
    printf("Select 1-3: \n");
    if(scanf("%d",&act3) != 1){
        perror("Error reading data from standard input");
    }
    
    if(act3 == 1){
        act_h3 = sigmoid;
    }else if(act3 == 2){
        act_h3 = relu;
    }else if(act3 == 3){
        act_h3 = tanh;
    }
    printf("---------------------------------------------\n");
    printf("Give me B: ");
    if (scanf("%d",&B) == 1) {
        while (TRAIN%B != 0){
            printf("Must be 4000  B = 0\n");
            printf("Give me again B: ");
            
            if(scanf("%d",&B) != 1){
                perror("Error reading data from standard input");
            }
        }
        printf("You gave me the %d\n",B);
    } else {
        printf("Invalid input\n");
    }

    if(B == 1){
        gradient_descent_serial(&input,&h1,&h2,&h3,&out,act_h1,act_h2,act_h3);
    }else{
        gradient_descent_team(&input,&h1,&h2,&h3,&out,act_h1,act_h2,act_h3,B);
        
    }
    test(&input,&h1,&h2,&h3,&out,&correct,&false);

    printf("Correct: %f\n",correct);
    printf("Rate: %f\n",((double)correct/TEST)*100);
    int result = system("python3 graphs.py");
    return 0;
}