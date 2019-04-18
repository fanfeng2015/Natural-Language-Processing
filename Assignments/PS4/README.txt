Shareable Link
---------------------------------------------------------------------------------------------------
https://colab.research.google.com/drive/1KjD2XrcP6pt4H5yEYNb89Glch8FDjGmd



Answers to Short Questions
---------------------------------------------------------------------------------------------------
1. 
Q: Why do we run model.eval() during the evaluation?
A: By default all modules in PyTorch are initialized to train mode, model.eval() explicitely indicates that code is switched to evaluation mode. Note that some layers may have different behaviors in train mode vs. evaluation mode, for example, bacth norm, or dropout.

2. 
Q: Why do we run optimizer.zero_grad() during the training?
A: PyTorch by default accumulates gradients on subsequent backward passes. This is useful when training RNN, but in some other occasions where we don't want to accumulate the gradients (e.g., training a regular feed forward NN with mini-batches), optimizer.zero_grad() should be run during training.

3.
Q: What do loss.backward() and optimizer.step() do?
A: loss.backward() computes the gradients (of the loss w.r.t. the parameters), optimizer.step() performs the gradient descent (to actually update the parameters).

4.
Q: Write a SQL query for "What is the name of the department with the highest average instructor salary?"
A:
   SELECT T2.name
   FROM instructor as T1 JOIN department as T2 ON T1.department_id = T2.id
   GROUP BY T1.department_id
   ORDER BY avg(T1.salary) DESC
   LIMIT 1

5.
Q: Now you implemented one module in SQLNet. List two limitations of SQLNet model and two possible ideas for the Spider task.
​A: 
   Limitations of SQLNet:
   1. SQLNet doesn't handle scenarios where nested SQL queries are needed. 
   2. The ORDER BY module considers only 0 or 1 column, but in reality multiple columns can be included in the ORDER BY clause. 

   Possible ideas for the Spider task:
   1. Build a diaglog interface that allows the system to interact with human to generate SQL queries that best suit human needs.
   2. Since queries are marked with difficulties, Spider can be used to compare the performance discrepancy of a particular model on queries of different difficulties. Or similarly, by comparing different models, Spider can be used to suggest the particular set of features that results in good/bad performance on queries of a specific level of difficulty.



Outputs (Part 2 & 3)
---------------------------------------------------------------------------------------------------
Part 2

Epoch 1 @ 2019-04-09 21:49:55.605207
 Loss = 7.2532005255562915
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:410: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
 Train acc_qm: 0.011285714285714286
 Breakdown results: sel: 0.08714285714285715, cond: 0.36842857142857144, group: 0.7508571428571429, order: 0.7304285714285714
 Dev acc_qm: 0.008704061895551257
 Breakdown results: sel: 0.10638297872340426, cond: 0.3413926499032882, group: 0.7350096711798839, order: 0.7079303675048356
Saving sel model...
Saving cond model...
Saving group model...
Saving order model...
 Best val sel = 0.10638297872340426, cond = 0.3413926499032882, group = 0.7350096711798839, order = 0.7079303675048356, tot = 0.008704061895551257
Epoch 2 @ 2019-04-09 21:52:20.139362
 Loss = 5.317157589503697
 Train acc_qm: 0.008285714285714285
 Breakdown results: sel: 0.09471428571428571, cond: 0.43257142857142855, group: 0.6728571428571428, order: 0.7961428571428572
 Dev acc_qm: 0.005802707930367505
 Breakdown results: sel: 0.11798839458413926, cond: 0.42940038684719534, group: 0.648936170212766, order: 0.7775628626692457
Saving sel model...
Saving cond model...
Saving order model...
 Best val sel = 0.11798839458413926, cond = 0.42940038684719534, group = 0.7350096711798839, order = 0.7775628626692457, tot = 0.008704061895551257
Epoch 3 @ 2019-04-09 21:54:46.204914
 Loss = 4.5341595279148645
 Train acc_qm: 0.03371428571428572
 Breakdown results: sel: 0.14985714285714286, cond: 0.449, group: 0.7091428571428572, order: 0.8104285714285714
 Dev acc_qm: 0.04448742746615087
 Breakdown results: sel: 0.21083172147001933, cond: 0.4574468085106383, group: 0.688588007736944, order: 0.8085106382978723
Saving sel model...
Saving cond model...
Saving order model...
 Best val sel = 0.21083172147001933, cond = 0.4574468085106383, group = 0.7350096711798839, order = 0.8085106382978723, tot = 0.04448742746615087
Epoch 4 @ 2019-04-09 21:57:10.310820
 Loss = 4.137586430413382
 Train acc_qm: 0.04371428571428571
 Breakdown results: sel: 0.17357142857142857, cond: 0.4288571428571429, group: 0.7342857142857143, order: 0.8271428571428572
 Dev acc_qm: 0.05125725338491296
 Breakdown results: sel: 0.2437137330754352, cond: 0.4284332688588008, group: 0.7156673114119922, order: 0.8239845261121856
Saving sel model...
Saving order model...
 Best val sel = 0.2437137330754352, cond = 0.4574468085106383, group = 0.7350096711798839, order = 0.8239845261121856, tot = 0.05125725338491296
Epoch 5 @ 2019-04-09 21:59:33.338166
 Loss = 3.820557077407837
 Train acc_qm: 0.046142857142857145
 Breakdown results: sel: 0.17942857142857144, cond: 0.48242857142857143, group: 0.718, order: 0.8265714285714286
 Dev acc_qm: 0.059961315280464215
 Breakdown results: sel: 0.2562862669245648, cond: 0.48452611218568664, group: 0.6827852998065764, order: 0.8249516441005803
Saving sel model...
Saving cond model...
Saving order model...
 Best val sel = 0.2562862669245648, cond = 0.48452611218568664, group = 0.7350096711798839, order = 0.8249516441005803, tot = 0.059961315280464215
Epoch 6 @ 2019-04-09 22:01:57.150905
 Loss = 3.6295712285723005
 Train acc_qm: 0.04757142857142857
 Breakdown results: sel: 0.187, cond: 0.4888571428571429, group: 0.7367142857142858, order: 0.8335714285714285
 Dev acc_qm: 0.06576402321083172
 Breakdown results: sel: 0.2640232108317215, cond: 0.4864603481624758, group: 0.7040618955512572, order: 0.8326885880077369
Saving sel model...
Saving cond model...
Saving order model...
 Best val sel = 0.2640232108317215, cond = 0.4864603481624758, group = 0.7350096711798839, order = 0.8326885880077369, tot = 0.06576402321083172
Epoch 7 @ 2019-04-09 22:04:22.326935
 Loss = 3.4739819483075824
 Train acc_qm: 0.052
 Breakdown results: sel: 0.18271428571428572, cond: 0.5117142857142857, group: 0.745, order: 0.8262857142857143
 Dev acc_qm: 0.06673114119922631
 Breakdown results: sel: 0.23984526112185686, cond: 0.5203094777562862, group: 0.7195357833655706, order: 0.804642166344294
Saving cond model...
 Best val sel = 0.2640232108317215, cond = 0.5203094777562862, group = 0.7350096711798839, order = 0.8326885880077369, tot = 0.06673114119922631
Epoch 8 @ 2019-04-09 22:06:46.254440
 Loss = 3.337534571783883
 Train acc_qm: 0.05442857142857143
 Breakdown results: sel: 0.18985714285714286, cond: 0.5378571428571428, group: 0.732, order: 0.8208571428571428
 Dev acc_qm: 0.0735009671179884
 Breakdown results: sel: 0.2620889748549323, cond: 0.558027079303675, group: 0.6992263056092843, order: 0.8017408123791102
Saving cond model...
 Best val sel = 0.2640232108317215, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8326885880077369, tot = 0.0735009671179884
Epoch 9 @ 2019-04-09 22:09:10.576580
 Loss = 3.1860861571175714
 Train acc_qm: 0.054285714285714284
 Breakdown results: sel: 0.19614285714285715, cond: 0.5258571428571429, group: 0.714, order: 0.8431428571428572
 Dev acc_qm: 0.07640232108317214
 Breakdown results: sel: 0.2688588007736944, cond: 0.5338491295938105, group: 0.6663442940038685, order: 0.8317214700193424
Saving sel model...
 Best val sel = 0.2688588007736944, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8326885880077369, tot = 0.07640232108317214
Epoch 10 @ 2019-04-09 22:11:33.805057
 Loss = 3.096678188596453
 Train acc_qm: 0.053285714285714283
 Breakdown results: sel: 0.19542857142857142, cond: 0.5055714285714286, group: 0.7068571428571429, order: 0.8432857142857143
 Dev acc_qm: 0.07059961315280464
 Breakdown results: sel: 0.2553191489361702, cond: 0.5029013539651838, group: 0.6499032882011605, order: 0.8317214700193424
 Best val sel = 0.2688588007736944, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8326885880077369, tot = 0.07640232108317214
Epoch 11 @ 2019-04-09 22:13:47.634882
 Loss = 2.9964893981388636
 Train acc_qm: 0.05542857142857143
 Breakdown results: sel: 0.19242857142857142, cond: 0.5285714285714286, group: 0.7362857142857143, order: 0.8494285714285714
 Dev acc_qm: 0.08220502901353965
 Breakdown results: sel: 0.2572533849129594, cond: 0.5290135396518375, group: 0.6818181818181818, order: 0.8413926499032882
Saving order model...
 Best val sel = 0.2688588007736944, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8413926499032882, tot = 0.08220502901353965
Epoch 12 @ 2019-04-09 22:16:12.968594
 Loss = 2.8900946494511195
 Train acc_qm: 0.054714285714285715
 Breakdown results: sel: 0.18528571428571428, cond: 0.49057142857142855, group: 0.747, order: 0.8537142857142858
 Dev acc_qm: 0.06479690522243714
 Breakdown results: sel: 0.2437137330754352, cond: 0.4622823984526112, group: 0.7117988394584139, order: 0.8520309477756286
Saving order model...
 Best val sel = 0.2688588007736944, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8520309477756286, tot = 0.08220502901353965
Epoch 13 @ 2019-04-09 22:18:38.982734
 Loss = 2.8795376126425607
 Train acc_qm: 0.05728571428571429
 Breakdown results: sel: 0.19614285714285715, cond: 0.5402857142857143, group: 0.7262857142857143, order: 0.8541428571428571
 Dev acc_qm: 0.08220502901353965
 Breakdown results: sel: 0.2553191489361702, cond: 0.5367504835589942, group: 0.6721470019342359, order: 0.8404255319148937
 Best val sel = 0.2688588007736944, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8520309477756286, tot = 0.08220502901353965
Epoch 14 @ 2019-04-09 22:21:04.609479
 Loss = 2.7134429234095983
 Train acc_qm: 0.05785714285714286
 Breakdown results: sel: 0.19842857142857143, cond: 0.5164285714285715, group: 0.732, order: 0.8631428571428571
 Dev acc_qm: 0.07930367504835589
 Breakdown results: sel: 0.2562862669245648, cond: 0.5, group: 0.6914893617021277, order: 0.8500967117988395
 Best val sel = 0.2688588007736944, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8520309477756286, tot = 0.08220502901353965
Epoch 15 @ 2019-04-09 22:23:29.951580
 Loss = 2.6879943190983364
 Train acc_qm: 0.05914285714285714
 Breakdown results: sel: 0.2057142857142857, cond: 0.5488571428571428, group: 0.6938571428571428, order: 0.8628571428571429
 Dev acc_qm: 0.0725338491295938
 Breakdown results: sel: 0.25918762088974856, cond: 0.5444874274661509, group: 0.6237911025145068, order: 0.8462282398452611
 Best val sel = 0.2688588007736944, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8520309477756286, tot = 0.08220502901353965
Epoch 16 @ 2019-04-09 22:25:53.353929
 Loss = 2.6244166870117187
 Train acc_qm: 0.06914285714285714
 Breakdown results: sel: 0.20885714285714285, cond: 0.5472857142857143, group: 0.7641428571428571, order: 0.8711428571428571
 Dev acc_qm: 0.08897485493230174
 Breakdown results: sel: 0.25822050290135395, cond: 0.5309477756286267, group: 0.7292069632495164, order: 0.8500967117988395
 Best val sel = 0.2688588007736944, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8520309477756286, tot = 0.08897485493230174
Epoch 17 @ 2019-04-09 22:28:13.217033
 Loss = 2.5423857007707866
 Train acc_qm: 0.06857142857142857
 Breakdown results: sel: 0.21014285714285713, cond: 0.5442857142857143, group: 0.753, order: 0.872
 Dev acc_qm: 0.08897485493230174
 Breakdown results: sel: 0.2727272727272727, cond: 0.5193423597678917, group: 0.7021276595744681, order: 0.8539651837524178
Saving sel model...
Saving order model...
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8539651837524178, tot = 0.08897485493230174
Epoch 18 @ 2019-04-09 22:30:38.011465
 Loss = 2.4935222832815986
 Train acc_qm: 0.06714285714285714
 Breakdown results: sel: 0.21042857142857144, cond: 0.5214285714285715, group: 0.7531428571428571, order: 0.8728571428571429
 Dev acc_qm: 0.08413926499032882
 Breakdown results: sel: 0.2678916827852998, cond: 0.4864603481624758, group: 0.7059961315280464, order: 0.8346228239845261
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8539651837524178, tot = 0.08897485493230174
Epoch 19 @ 2019-04-09 22:33:02.756890
 Loss = 2.452490427562169
 Train acc_qm: 0.07228571428571429
 Breakdown results: sel: 0.213, cond: 0.5542857142857143, group: 0.7631428571428571, order: 0.8781428571428571
 Dev acc_qm: 0.08994197292069632
 Breakdown results: sel: 0.269825918762089, cond: 0.5386847195357833, group: 0.718568665377176, order: 0.8520309477756286
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8539651837524178, tot = 0.08994197292069632
Epoch 20 @ 2019-04-09 22:35:25.063858
 Loss = 2.4088190081460135
 Train acc_qm: 0.06942857142857142
 Breakdown results: sel: 0.2092857142857143, cond: 0.5515714285714286, group: 0.7652857142857142, order: 0.8778571428571429
 Dev acc_qm: 0.09090909090909091
 Breakdown results: sel: 0.2572533849129594, cond: 0.5386847195357833, group: 0.7137330754352031, order: 0.8491295938104448
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8539651837524178, tot = 0.09090909090909091
Epoch 21 @ 2019-04-09 22:37:48.767081
 Loss = 2.3290985785893032
 Train acc_qm: 0.07271428571428572
 Breakdown results: sel: 0.219, cond: 0.5395714285714286, group: 0.7602857142857142, order: 0.8807142857142857
 Dev acc_qm: 0.0851063829787234
 Breakdown results: sel: 0.2601547388781431, cond: 0.4912959381044487, group: 0.7040618955512572, order: 0.8558994197292069
Saving order model...
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8558994197292069, tot = 0.09090909090909091
Epoch 22 @ 2019-04-09 22:40:13.760904
 Loss = 2.289359140941075
 Train acc_qm: 0.07642857142857143
 Breakdown results: sel: 0.21771428571428572, cond: 0.5557142857142857, group: 0.7567142857142857, order: 0.8825714285714286
 Dev acc_qm: 0.08800773694390715
 Breakdown results: sel: 0.2572533849129594, cond: 0.5164410058027079, group: 0.6943907156673114, order: 0.8481624758220503
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8558994197292069, tot = 0.09090909090909091
Epoch 23 @ 2019-04-09 22:42:37.212805
 Loss = 2.309070537022182
 Train acc_qm: 0.07685714285714286
 Breakdown results: sel: 0.21971428571428572, cond: 0.5641428571428572, group: 0.752, order: 0.8804285714285714
 Dev acc_qm: 0.08994197292069632
 Breakdown results: sel: 0.26595744680851063, cond: 0.5377176015473888, group: 0.6876208897485493, order: 0.8549323017408124
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8558994197292069, tot = 0.09090909090909091
Epoch 24 @ 2019-04-09 22:45:00.470158
 Loss = 2.25143064226423
 Train acc_qm: 0.08028571428571428
 Breakdown results: sel: 0.21771428571428572, cond: 0.5625714285714286, group: 0.7695714285714286, order: 0.8864285714285715
 Dev acc_qm: 0.08317214700193423
 Breakdown results: sel: 0.26595744680851063, cond: 0.534816247582205, group: 0.7340425531914894, order: 0.8491295938104448
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8558994197292069, tot = 0.09090909090909091
Epoch 25 @ 2019-04-09 22:47:22.361226
 Loss = 2.189972674506051
 Train acc_qm: 0.08242857142857143
 Breakdown results: sel: 0.22457142857142856, cond: 0.554, group: 0.7672857142857142, order: 0.8864285714285715
 Dev acc_qm: 0.0851063829787234
 Breakdown results: sel: 0.2620889748549323, cond: 0.511605415860735, group: 0.718568665377176, order: 0.8578336557059961
Saving order model...
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8578336557059961, tot = 0.09090909090909091
Epoch 26 @ 2019-04-09 22:49:45.890512
 Loss = 2.177378126961844
 Train acc_qm: 0.08228571428571428
 Breakdown results: sel: 0.22685714285714287, cond: 0.5718571428571428, group: 0.7661428571428571, order: 0.8851428571428571
 Dev acc_qm: 0.08897485493230174
 Breakdown results: sel: 0.2640232108317215, cond: 0.5570599613152805, group: 0.7108317214700194, order: 0.8404255319148937
 Best val sel = 0.2727272727272727, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8578336557059961, tot = 0.09090909090909091
Epoch 27 @ 2019-04-09 22:52:01.049121
 Loss = 2.1684843087877548
 Train acc_qm: 0.08471428571428571
 Breakdown results: sel: 0.22757142857142856, cond: 0.5762857142857143, group: 0.7692857142857142, order: 0.8884285714285715
 Dev acc_qm: 0.09284332688588008
 Breakdown results: sel: 0.2804642166344294, cond: 0.5377176015473888, group: 0.7321083172147002, order: 0.839458413926499
Saving sel model...
 Best val sel = 0.2804642166344294, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8578336557059961, tot = 0.09284332688588008
Epoch 28 @ 2019-04-09 22:54:23.080779
 Loss = 2.1235083359309606
 Train acc_qm: 0.086
 Breakdown results: sel: 0.22528571428571428, cond: 0.5701428571428572, group: 0.7725714285714286, order: 0.8908571428571429
 Dev acc_qm: 0.09187620889748549
 Breakdown results: sel: 0.27079303675048355, cond: 0.5261121856866537, group: 0.7263056092843327, order: 0.8568665377176016
 Best val sel = 0.2804642166344294, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8578336557059961, tot = 0.09284332688588008
Epoch 29 @ 2019-04-09 22:56:46.882927
 Loss = 2.088980818203517
 Train acc_qm: 0.08614285714285715
 Breakdown results: sel: 0.22785714285714287, cond: 0.5794285714285714, group: 0.7674285714285715, order: 0.8898571428571429
 Dev acc_qm: 0.09381044487427466
 Breakdown results: sel: 0.27176015473887816, cond: 0.5483558994197292, group: 0.7040618955512572, order: 0.8529980657640233
 Best val sel = 0.2804642166344294, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8578336557059961, tot = 0.09381044487427466
Epoch 30 @ 2019-04-09 22:59:10.205667
 Loss = 2.0430272545133317
 Train acc_qm: 0.086
 Breakdown results: sel: 0.2317142857142857, cond: 0.5784285714285714, group: 0.7725714285714286, order: 0.888
 Dev acc_qm: 0.08800773694390715
 Breakdown results: sel: 0.27176015473887816, cond: 0.5483558994197292, group: 0.723404255319149, order: 0.8462282398452611
 Best val sel = 0.2804642166344294, cond = 0.558027079303675, group = 0.7350096711798839, order = 0.8578336557059961, tot = 0.09381044487427466



Part 3

Epoch 1 @ 2019-04-09 23:06:53.816101
 Loss = 1.6299720948392695
 Train acc_qm: 0.451
 Breakdown results: sel: 1.0, cond: 0.451, group: 1.0, order: 1.0
 Dev acc_qm: 0.4284332688588008
 Breakdown results: sel: 1.0, cond: 0.4284332688588008, group: 1.0, order: 1.0
Saving cond model...
 Best val sel = 1.0, cond = 0.4284332688588008, group = 1.0, order = 1.0, tot = 0.4284332688588008
Epoch 2 @ 2019-04-09 23:07:29.315893
 Loss = 1.2228886967355554
 Train acc_qm: 0.4677142857142857
 Breakdown results: sel: 1.0, cond: 0.4677142857142857, group: 1.0, order: 1.0
 Dev acc_qm: 0.45551257253384914
 Breakdown results: sel: 1.0, cond: 0.45551257253384914, group: 1.0, order: 1.0
Saving cond model...
 Best val sel = 1.0, cond = 0.45551257253384914, group = 1.0, order = 1.0, tot = 0.45551257253384914
Epoch 3 @ 2019-04-09 23:08:05.989370
 Loss = 0.9800558380105279
 Train acc_qm: 0.5031428571428571
 Breakdown results: sel: 1.0, cond: 0.5031428571428571, group: 1.0, order: 1.0
 Dev acc_qm: 0.488394584139265
 Breakdown results: sel: 1.0, cond: 0.488394584139265, group: 1.0, order: 1.0
Saving cond model...
 Best val sel = 1.0, cond = 0.488394584139265, group = 1.0, order = 1.0, tot = 0.488394584139265
Epoch 4 @ 2019-04-09 23:08:45.589590
 Loss = 0.8015877550298517
 Train acc_qm: 0.4908571428571429
 Breakdown results: sel: 1.0, cond: 0.4908571428571429, group: 1.0, order: 1.0
 Dev acc_qm: 0.43617021276595747
 Breakdown results: sel: 1.0, cond: 0.43617021276595747, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.488394584139265, group = 1.0, order = 1.0, tot = 0.488394584139265
Epoch 5 @ 2019-04-09 23:09:20.062205
 Loss = 0.6890134857459502
 Train acc_qm: 0.5252857142857142
 Breakdown results: sel: 1.0, cond: 0.5252857142857142, group: 1.0, order: 1.0
 Dev acc_qm: 0.4941972920696325
 Breakdown results: sel: 1.0, cond: 0.4941972920696325, group: 1.0, order: 1.0
Saving cond model...
 Best val sel = 1.0, cond = 0.4941972920696325, group = 1.0, order = 1.0, tot = 0.4941972920696325
Epoch 6 @ 2019-04-09 23:09:53.472789
 Loss = 0.5944271301681345
 Train acc_qm: 0.5055714285714286
 Breakdown results: sel: 1.0, cond: 0.5055714285714286, group: 1.0, order: 1.0
 Dev acc_qm: 0.4497098646034816
 Breakdown results: sel: 1.0, cond: 0.4497098646034816, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.4941972920696325, group = 1.0, order = 1.0, tot = 0.4941972920696325
Epoch 7 @ 2019-04-09 23:10:30.747044
 Loss = 0.5115291935476389
 Train acc_qm: 0.539
 Breakdown results: sel: 1.0, cond: 0.539, group: 1.0, order: 1.0
 Dev acc_qm: 0.4632495164410058
 Breakdown results: sel: 1.0, cond: 0.4632495164410058, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.4941972920696325, group = 1.0, order = 1.0, tot = 0.4941972920696325
Epoch 8 @ 2019-04-09 23:11:05.984962
 Loss = 0.4660454653880813
 Train acc_qm: 0.5508571428571428
 Breakdown results: sel: 1.0, cond: 0.5508571428571428, group: 1.0, order: 1.0
 Dev acc_qm: 0.5164410058027079
 Breakdown results: sel: 1.0, cond: 0.5164410058027079, group: 1.0, order: 1.0
Saving cond model...
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 9 @ 2019-04-09 23:11:39.504274
 Loss = 0.41105934273112904
 Train acc_qm: 0.553
 Breakdown results: sel: 1.0, cond: 0.553, group: 1.0, order: 1.0
 Dev acc_qm: 0.48549323017408125
 Breakdown results: sel: 1.0, cond: 0.48549323017408125, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 10 @ 2019-04-09 23:12:13.468657
 Loss = 0.35107766457579354
 Train acc_qm: 0.5508571428571428
 Breakdown results: sel: 1.0, cond: 0.5508571428571428, group: 1.0, order: 1.0
 Dev acc_qm: 0.45357833655706
 Breakdown results: sel: 1.0, cond: 0.45357833655706, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 11 @ 2019-04-09 23:12:49.390507
 Loss = 0.3157094577496702
 Train acc_qm: 0.5714285714285714
 Breakdown results: sel: 1.0, cond: 0.5714285714285714, group: 1.0, order: 1.0
 Dev acc_qm: 0.4961315280464217
 Breakdown results: sel: 1.0, cond: 0.4961315280464217, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 12 @ 2019-04-09 23:13:25.891132
 Loss = 0.28595962910489603
 Train acc_qm: 0.5717142857142857
 Breakdown results: sel: 1.0, cond: 0.5717142857142857, group: 1.0, order: 1.0
 Dev acc_qm: 0.4796905222437137
 Breakdown results: sel: 1.0, cond: 0.4796905222437137, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 13 @ 2019-04-09 23:13:58.792767
 Loss = 0.2633158468387344
 Train acc_qm: 0.5651428571428572
 Breakdown results: sel: 1.0, cond: 0.5651428571428572, group: 1.0, order: 1.0
 Dev acc_qm: 0.46421663442940037
 Breakdown results: sel: 1.0, cond: 0.46421663442940037, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 14 @ 2019-04-09 23:14:36.115118
 Loss = 0.24034510986371474
 Train acc_qm: 0.5844285714285714
 Breakdown results: sel: 1.0, cond: 0.5844285714285714, group: 1.0, order: 1.0
 Dev acc_qm: 0.48549323017408125
 Breakdown results: sel: 1.0, cond: 0.48549323017408125, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 15 @ 2019-04-09 23:15:13.736606
 Loss = 0.2076548949561336
 Train acc_qm: 0.5844285714285714
 Breakdown results: sel: 1.0, cond: 0.5844285714285714, group: 1.0, order: 1.0
 Dev acc_qm: 0.4825918762088975
 Breakdown results: sel: 1.0, cond: 0.4825918762088975, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 16 @ 2019-04-09 23:15:48.001821
 Loss = 0.20878803431987764
 Train acc_qm: 0.5901428571428572
 Breakdown results: sel: 1.0, cond: 0.5901428571428572, group: 1.0, order: 1.0
 Dev acc_qm: 0.5096711798839458
 Breakdown results: sel: 1.0, cond: 0.5096711798839458, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 17 @ 2019-04-09 23:16:22.056218
 Loss = 0.18364378068257461
 Train acc_qm: 0.5985714285714285
 Breakdown results: sel: 1.0, cond: 0.5985714285714285, group: 1.0, order: 1.0
 Dev acc_qm: 0.5106382978723404
 Breakdown results: sel: 1.0, cond: 0.5106382978723404, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 18 @ 2019-04-09 23:16:58.934904
 Loss = 0.15556318566880442
 Train acc_qm: 0.5985714285714285
 Breakdown results: sel: 1.0, cond: 0.5985714285714285, group: 1.0, order: 1.0
 Dev acc_qm: 0.5009671179883946
 Breakdown results: sel: 1.0, cond: 0.5009671179883946, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 19 @ 2019-04-09 23:17:33.604766
 Loss = 0.14956041401760145
 Train acc_qm: 0.594
 Breakdown results: sel: 1.0, cond: 0.594, group: 1.0, order: 1.0
 Dev acc_qm: 0.4796905222437137
 Breakdown results: sel: 1.0, cond: 0.4796905222437137, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5164410058027079, group = 1.0, order = 1.0, tot = 0.5164410058027079
Epoch 20 @ 2019-04-09 23:18:07.243060
 Loss = 0.14819020351225679
 Train acc_qm: 0.6044285714285714
 Breakdown results: sel: 1.0, cond: 0.6044285714285714, group: 1.0, order: 1.0
 Dev acc_qm: 0.5319148936170213
 Breakdown results: sel: 1.0, cond: 0.5319148936170213, group: 1.0, order: 1.0
Saving cond model...
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 21 @ 2019-04-09 23:18:44.226413
 Loss = 0.14263986992565067
 Train acc_qm: 0.6027142857142858
 Breakdown results: sel: 1.0, cond: 0.6027142857142858, group: 1.0, order: 1.0
 Dev acc_qm: 0.5019342359767892
 Breakdown results: sel: 1.0, cond: 0.5019342359767892, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 22 @ 2019-04-09 23:19:19.441752
 Loss = 0.12060600700364872
 Train acc_qm: 0.6064285714285714
 Breakdown results: sel: 1.0, cond: 0.6064285714285714, group: 1.0, order: 1.0
 Dev acc_qm: 0.5077369439071566
 Breakdown results: sel: 1.0, cond: 0.5077369439071566, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 23 @ 2019-04-09 23:19:52.706503
 Loss = 0.11311200384727933
 Train acc_qm: 0.6091428571428571
 Breakdown results: sel: 1.0, cond: 0.6091428571428571, group: 1.0, order: 1.0
 Dev acc_qm: 0.5222437137330754
 Breakdown results: sel: 1.0, cond: 0.5222437137330754, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 24 @ 2019-04-09 23:20:25.984042
 Loss = 0.11894066696139899
 Train acc_qm: 0.608
 Breakdown results: sel: 1.0, cond: 0.608, group: 1.0, order: 1.0
 Dev acc_qm: 0.5019342359767892
 Breakdown results: sel: 1.0, cond: 0.5019342359767892, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 25 @ 2019-04-09 23:21:01.366779
 Loss = 0.10459769310599024
 Train acc_qm: 0.6101428571428571
 Breakdown results: sel: 1.0, cond: 0.6101428571428571, group: 1.0, order: 1.0
 Dev acc_qm: 0.5164410058027079
 Breakdown results: sel: 1.0, cond: 0.5164410058027079, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 26 @ 2019-04-09 23:21:37.655868
 Loss = 0.10622279268096793
 Train acc_qm: 0.6101428571428571
 Breakdown results: sel: 1.0, cond: 0.6101428571428571, group: 1.0, order: 1.0
 Dev acc_qm: 0.5077369439071566
 Breakdown results: sel: 1.0, cond: 0.5077369439071566, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 27 @ 2019-04-09 23:22:14.770546
 Loss = 0.11214756264605305
 Train acc_qm: 0.6025714285714285
 Breakdown results: sel: 1.0, cond: 0.6025714285714285, group: 1.0, order: 1.0
 Dev acc_qm: 0.4961315280464217
 Breakdown results: sel: 1.0, cond: 0.4961315280464217, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 28 @ 2019-04-09 23:22:53.032182
 Loss = 0.1328181087293408
 Train acc_qm: 0.61
 Breakdown results: sel: 1.0, cond: 0.61, group: 1.0, order: 1.0
 Dev acc_qm: 0.5164410058027079
 Breakdown results: sel: 1.0, cond: 0.5164410058027079, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 29 @ 2019-04-09 23:23:29.339405
 Loss = 0.10563933376900174
 Train acc_qm: 0.6174285714285714
 Breakdown results: sel: 1.0, cond: 0.6174285714285714, group: 1.0, order: 1.0
 Dev acc_qm: 0.5251450676982592
 Breakdown results: sel: 1.0, cond: 0.5251450676982592, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213
Epoch 30 @ 2019-04-09 23:24:04.119532
 Loss = 0.10035321304405277
 Train acc_qm: 0.621
 Breakdown results: sel: 1.0, cond: 0.621, group: 1.0, order: 1.0
 Dev acc_qm: 0.5222437137330754
 Breakdown results: sel: 1.0, cond: 0.5222437137330754, group: 1.0, order: 1.0
 Best val sel = 1.0, cond = 0.5319148936170213, group = 1.0, order = 1.0, tot = 0.5319148936170213


