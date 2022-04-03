df1 <- read.csv("df1.csv")
head(df1)

elasticity = df1$elasticity
current_rate = df1$Current_Price
l12_sales_vol = df1$l12_sales_vol
cost = df1$standard_cost

pinc_list <- list()
pinc_cons_list <- list()

for (row in 1:nrow(df1)) {

    i = 0

  f3 <- function(pinc) {

    i = i+1

#     print(cat(i,' - ',pinc))

    new_vol = l12_sales_vol[row] * (1+(elasticity[row]*pinc))
    new_rev = new_vol * (current_rate[row] * (1+pinc))
    newcost = new_vol * cost[row]

    return(new_rev-newcost)
  }


    change_inv <- function(frac) {  l12_sales_vol[row] * (1 + frac) }
    n_vol_inv <- function(vol) {  (vol / l12_sales_vol[row] - 1) / elasticity[row] }
    n_vol_inv(vol = change_inv(frac = -0.1))


    res_un <- optimize(f3,lower=0, upper=10, maximum = TRUE)
    res <- optimize(f3,lower=0, upper=n_vol_inv(vol = change_inv(-0.1)), maximum = TRUE)



    pinc_cons_list[[row]] <- res$maximum
    pinc_list[[row]] <- res_un$maximum


}



df1$pinc = pinc_list
df1$pinc_constrained = pinc_cons_list
df1 <- apply(df1,2,as.character)

write.csv(df1,'Price_Opti_Output.csv')
