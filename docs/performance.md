# Bill Dusch's Instagram ML models 
## Performance of models
* First model series: labels only
* Second model series: labels + hour + date variables

### Linear
#### Labels Only
* kissinfashion: 4602.57502528 MAE ~ 57/202 points
* instagood: 2087.99913054 MAE ~ 87/202 points
* beautifuldestinations: 33277.1234229 MAE ~ 79/202 points
* etdieucrea: 453.45723004 MAE ~ 109/202 points
* josecabaco: 131.935522322 MAE ~ 69/202 points

#### Labels + Time
* kissinfashion: 4520.81188119 MAE ~ 60/202 points
* instagood: 2088.35643564 MAE ~ 86/202 points
* beautifuldestinations: 33587.6435644 MAE ~ 83/202 points
* etdieucrea: 458.896039604 MAE ~ 109/202 points
* josecabaco: 134.014851485 MAE ~ 69/202 points


### Ridge
#### Labels Only
* kissinfashion: 4142.25573798 MAE ~ 64/202 points
* instagood: 1816.72181263 MAE ~ 95/202 points
* beautifuldestinations: 29704.204573 MAE ~ 73/202 points
* etdieucrea: 404.257893164 MAE ~ 116/202 points
* josecabaco: 116.1684068 MAE ~ 77/202 points

#### Labels + Time
* kissinfashion: 4112.18383624 MAE ~ 67/202 points
* instagood: 1845.4208167 MAE ~ 94/202 points
* beautifuldestinations: 29729.7336933 MAE ~ 67/202 points
* etdieucrea: 401.785995813 MAE ~ 118/202 points
* josecabaco: 116.854176079 MAE ~ 78/202 points

### Lasso
#### Labels Only
* kissinfashion: 4176.52478816 MAE ~ 65/202 points
* instagood: 1821.91547773 MAE ~ 98/202 points
* beautifuldestinations: 29575.2704578 MAE ~ 76/202 points
* etdieucrea: 405.023896461 MAE ~ 122/202 points
* josecabaco: 116.152360358 MAE ~ 75/202 points

#### Labels + Time
* kissinfashion: 4157.52850314 MAE ~ 63/202 points
* instagood: 1841.33991476 MAE ~ 96/202 points
* beautifuldestinations: 30389.8472594 MAE ~ 65/202 points
* etdieucrea: 401.405645914 MAE ~ 122/202 points
* josecabaco: 116.67074797 MAE ~ 78/202 points


### Kernel Ridge
#### Labels Only
* kissinfashion: 4140.0868502 MAE ~ 64/202 points
* instagood: 1817.33653531 MAE ~ 99/202 points
* beautifuldestinations: 29679.0903986 MAE ~ 75/202 points
* etdieucrea: 404.235726669 MAE ~ 122/202 points
* josecabaco: 116.177468668 MAE ~ 79/202 points

#### Labels + Time
* kissinfashion: 4110.57801598 MAE ~ 67/202 points
* instagood: 1852.42063023 MAE ~ 93/202 points
* beautifuldestinations: 29683.0691751 MAE ~ 67/202 points
* etdieucrea: 401.187860321 MAE ~ 122/202 points
* josecabaco: 116.177530226 MAE ~ 79/202 points

### Support Vector Machine Regressor
#### Labels Only
* kissinfashion: 4156.72564171 MAE ~ 62/202 points
* instagood: 1915.69057442 MAE ~ 101/202 points
* beautifuldestinations: 30819.6865347 MAE ~ 73/202 points
* etdieucrea: 400.5017613 MAE ~ 124/202 points
* josecabaco: 113.583308551 MAE ~ 77/202 points

#### Labels + Time
* kissinfashion: 4129.06994594 MAE ~ 65/202 points
* instagood: 1933.3969209 MAE ~ 104/202 points
* beautifuldestinations: 30826.8349817 MAE ~ 73/202 points
* etdieucrea: 402.441779542 MAE ~ 122/202 points
* josecabaco: 113.484982268 MAE ~ 76/202 points

### Random Forest Regressor
#### Labels Only
* kissinfashion: 4150.51052915 MAE ~ 65/202 points
* instagood: 1849.58839717 MAE ~ 93/202 points
* beautifuldestinations: 29514.5957074 MAE ~ 67/202 points
* etdieucrea: 406.166659751 MAE ~ 120/202 points
* josecabaco: 115.911653447 MAE ~ 73/202 points

#### Labels + Time


## Selected Models
### Labels Only
* kissinfashion kernelridge: ~ {'alpha': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}
* instagood: ridge ~ {'alpha': 15}
* beautifuldestinations: randomforest ~ {'max_features': 'log2', 'min_samples_split': 2, 'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 1}
* etdieucrea: svr ~ {'kernel': 'rbf', 'C': 1000, 'gamma': 'auto'}
* josecabaco: svr ~ {'kernel': 'rbf', 'C': 100, 'gamma': 0.01}

### Labels + Time
