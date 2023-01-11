#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import numpy as np

import asyncio

CONTRACTS = ["LBSJ","LBSM", "LBSQ", "LBSV", "LBSZ"]
ORDER_SIZE = 20
SPREAD = 2
CONTRACT_TO_DAY = {
    "LBSJ": 4 * 21 - 1,
    "LBSM": 6 * 21 - 1,
    "LBSQ": 8 * 21 - 1, 
    "LBSV": 10 * 21 - 1,
    "LBSZ": 12 * 21 - 1
}

lst = [318.95563750428664, 317.80537346793056, 316.9670596647063, 316.40513632893783, 316.0840436949492, 315.9682219970645, 316.02211146960803, 316.21015234690367, 316.4967848632757, 316.8464492530481, 317.2235857505449, 317.59412292750596, 317.9299427053333, 318.2044153428449, 318.3909110988581, 318.46280023219094, 318.3858546611099, 318.09545294167617, 317.5193752894, 316.5854019197913, 315.2213130483603, 313.4227124406555, 311.45649806237964, 309.6573914292738, 308.360114057079, 307.89938746153655, 308.46300029569903, 309.65100976186477, 310.91654019964346, 311.71271594864504, 311.49266134847915, 309.89529011893353, 307.30267350050707, 304.28267211387634, 301.403146579718, 299.2319575187089, 298.1901267755302, 298.11132109088135, 298.68236842946607, 299.5900967559884, 300.5213340351523, 301.2243462270018, 301.6931512729417, 301.98320510971666, 302.1499636740717, 302.24888290275186, 302.3092144447793, 302.2553927982864, 301.9856481736827, 301.3982107813779, 300.3913108317818, 298.9040399756692, 297.038935625275, 294.93939663319895, 292.7488218520411, 290.6106101344013, 288.65920750409407, 286.9932486697914, 285.7024155113794, 284.87638990874433, 284.6048537417729, 284.9179097443853, 285.6073440666389, 286.4053637126246, 287.04417568643373, 287.25598699215755, 286.86483272428825, 286.062060338922, 285.1308453825556, 284.35436340168627, 284.0157899428111, 284.2998148925008, 284.9971854976209, 285.80016334511043, 286.4010100219085, 286.4919871149545, 285.86827785643675, 284.7367520595413, 283.40720118270303, 282.1894166843573, 281.3931900229393, 281.2271414840019, 281.4952066615684, 281.90014997677946, 282.14473585077616, 281.9317287046995, 281.05863952299865, 279.70196554335644, 278.1329505667639, 276.62283839421224, 275.44287282669274, 274.8003311274604, 274.64662440882614, 274.8691972453646, 275.35549421165064, 275.992959882259, 276.6795149562786, 277.35498463085446, 277.96967022764557, 278.47387306831087, 278.81789447450967, 278.96605924926604, 278.9387861210651, 278.77051729975665, 278.495694995191, 278.1487614172183, 277.76410915553555, 277.3759323192278, 277.01837539722703, 276.7255828784651, 276.5316992518742, 276.4556079693689, 276.45514833479405, 276.4728986149771, 276.45143707674544, 276.3333419869266, 276.0917035963221, 275.82166009162984, 275.6488616435217, 275.69895842266953, 276.0976005997454, 276.91292252731915, 277.98299528555253, 279.088374136505, 280.0096143422361, 280.5272711648056, 280.4922439009838, 280.0368079863842, 279.36358289133085, 278.675188086148, 278.17424304116014, 278.01005231483487, 278.11866081821495, 278.3827985504865, 278.6851955108358, 278.9085816984492, 278.97453997127667, 278.9600646223234, 278.9810028033582, 279.15320166614987, 279.5925083624675, 280.3661197508548, 281.34663151695514, 282.35798905318654, 283.2241377519673, 283.7690230057156, 283.8734565810634, 283.6457157414974, 283.25094412471793, 282.8542853684254, 282.6208831103205, 282.67236816903664, 282.9563200869408, 283.3768055873329, 283.83789139351285, 284.2436442287808, 284.52598010438396, 284.72821218335815, 284.92150291668634, 285.17701475535154, 285.5659101503369, 286.13121601629877, 286.80341712258667, 287.4848627022235, 288.0779019882321, 288.4848842136356, 288.6550612189832, 288.72529527492935, 288.8793512596548, 289.30099405134047, 290.17398852816723, 291.6077453518351, 293.4142583181215, 295.3311670063229, 297.0961109957362, 298.4467298656577, 299.20034352961875, 299.49299323808935, 299.5404005757736, 299.55828712737593, 299.76237447760104, 300.32139648275927, 301.21613608558556, 302.38038850042113, 303.7479489416069, 305.2526126234842, 306.81918202041203, 308.33648864682135, 309.68437127716135, 310.74266868588074, 311.39121964742884, 311.5550470992996, 311.33991063116696, 310.8967539957498, 310.3765209457671, 309.930155233938, 309.6887170893264, 309.70373264637624, 310.0068445158765, 310.6296953086159, 311.6039276353837, 312.92828061548147, 314.46987940226256, 316.0629456575925, 317.54170104333747, 318.74036722136333, 319.54620506218697, 320.058632270929, 320.4301057613609, 320.813082447254, 321.36001924237996, 322.17610451956386, 323.17745248784524, 324.23290881531705, 325.21131917007233, 325.98152922020444, 326.44985035326636, 326.67245683465245, 326.7429886492167, 326.75508578181325, 326.8023882172966, 326.9554296560133, 327.19231866027997, 327.46805750790526, 327.7376484766982, 327.95609384446794, 328.08255445964886, 328.0928254531783, 327.9668605266192, 327.68461338153435, 327.22603771948684, 326.5916717753882, 325.8643919175456, 325.14765904761475, 324.54493406725123, 324.1596778781111, 324.06468348023657, 324.210072267216, 324.51529773102436, 324.89981336363644, 325.2830726570274, 325.60729312269984, 325.9057483502687, 326.2344759488765, 326.64951352766565, 327.2068986957791, 327.93268259462764, 328.73297049469613, 329.4838811987373, 330.0615335095043, 330.3420462297501, 330.2459029937327, 329.8710467617299, 329.35978532552406, 328.85442647689786, 328.4972780076341, 328.3976832427929, 328.53312764054493, 328.8481321923384, 329.28721788962145, 329.7949057238426, 330.31974606233496, 330.82640677597163, 331.28358511151043, 331.6599783157094, 331.9242836353267, 332.0360354414482, 331.91811660247214, 331.48424711112466, 330.6481469601317, 329.3235361422197, 327.49754694969766, 325.45096087320655, 323.53797170296997, 322.1127732292119, 321.5295592421562, 321.9928128170705, 323.10817416939756, 324.33157279962387, 325.1189382082361, 324.92619989572086, 323.39215686929487, 320.88708616309594, 317.96413431799203, 315.1764478748512, 313.0771733745414, 312.0756285841546, 312.00581617567735, 312.55791004732026, 313.4220840972938, 314.28851222380854, 314.909986805888, 315.2897741458075, 315.4937590266549, 315.5878262315186, 315.6378605434867, 315.68074747906377, 315.63737548842016, 315.3996337551424, 314.8594114628171, 313.90859779503114, 312.48002549789305, 310.67030156760035, 308.61697656287197, 306.45760104242737, 304.329725564986, 302.36408612064747, 300.6641604250327, 299.32661162514324, 298.4481028679803, 298.1252973005455, 298.39444629874237, 299.0501541540837, 299.82661338698455, 300.45801651785985, 300.6785560671248, 300.3128933442795, 299.54756481516426, 298.65957573470445, 297.92593135782556, 297.6236369394531, 297.93235027282566, 298.64233930443464, 299.4465245190845, 300.0378264015799, 300.1091654367255, 299.45703145066864, 298.2921916349281, 296.92898252236506, 295.68174064584116, 294.864802538218, 294.6902172497178, 294.9608839000067, 295.37741412611166, 295.6404195650595, 295.4505118538775, 294.6029485448426, 293.27157085123247, 291.72486590157473, 290.2313208243972, 289.0594227482276, 288.41459592959325, 288.25001313701983, 288.4557842670324, 288.9220192161561, 289.5388278809162, 290.20654921288497, 290.8664383838235, 291.46997962053996, 291.96865714984244, 292.3139551985393, 292.4707684483041, 292.45763340027224, 292.30649701044416, 292.0493062348205, 291.71800802940214, 291.3449123172636, 290.96378088977605, 290.6087385053844, 290.31390992253375, 290.1134198996692, 290.0264597713425, 290.0124871765311, 290.01602633031933, 289.9816014477911, 289.8537367440308, 289.6070299311302, 289.33637270921145, 289.1667302754042, 289.22306782683796, 289.63035056064274, 290.45593120432324, 291.5367126068847, 292.6519851477073, 293.58103920617145, 294.10316516165756, 294.06837282849057, 293.60954976077517, 292.9303029475605, 292.2342393778958, 291.72496604083057, 291.5527149319144, 291.6542180726986, 291.9128324912344, 292.2119152155736, 292.4348232737678, 292.5034965680499, 292.4942064973786, 292.52180733489394, 292.70115335373583, 293.14709882704454, 293.92599160734073, 294.9101538646674, 295.9234013484483, 296.7895498081071, 297.33241499306774, 297.4328369554179, 297.1997529579015, 296.7991245659265, 296.39691334490084, 296.1590808602326, 296.2079072402833, 296.4909468652291, 296.9120726781996, 297.37515762232454, 297.7840746407337, 298.07048399251556, 298.27719520059384, 298.474805103851, 298.7339105411693, 299.1251083514314, 299.6910131325441, 300.3623105185121, 301.0417039023644, 301.63189667713016, 302.0355922358384, 302.2023895294634, 302.26946974075906, 302.4209096104245, 302.8407858791587, 303.7131752876609, 305.14768381783034, 306.9560344163682, 308.8754792711756, 310.64327057015373, 311.99666050120413, 312.752629020811, 313.0470671597914, 313.0955937175449, 313.1138274934714, 313.31738728697087, 313.87497762214855, 314.7676459219308, 315.9295253339493, 317.2947490058356, 318.7974500852217, 320.3627056875838, 321.8793687997772, 323.2272363765013, 324.28610537245606, 324.9357727423413, 325.10118533191115, 324.8878895511369, 324.446581701044, 323.92795808265817, 323.482714997005, 323.24172672585536, 323.25657947396064, 323.55903742681727, 324.18086476992175, 325.1538256887709, 326.47678535689613, 328.01701289997055, 329.6088784317019, 331.0867520657981, 332.28500391596737, 333.0909939149113, 333.6040412713067, 333.9764550128241, 334.3605441671341, 334.9086177619072, 335.7257305856457, 336.7279204701788, 337.78397100816755, 338.7626657922728, 339.5327884151556, 340.0006214060593, 340.2224430405569, 340.2920305308039, 340.30316108895556, 340.3496119271675, 340.50203090441283, 340.73854846693604, 341.0141657077991, 341.2838837200644, 341.50270359679416, 341.62976750375265, 341.64078189751285, 341.51559430734966, 341.2340522625383, 340.77600329235383, 340.14190367896117, 339.41464471608367, 338.6977264503347, 338.0946489283274, 337.708912196675, 337.61335327321376, 337.75815706067135, 338.0628454329983, 338.4469402641452, 338.8299634280628, 339.1541802500348, 339.4528298606784, 339.78189484194354, 340.1973577757806, 340.7552012441398, 341.4814249810565, 342.2820973289065, 343.03330378215054, 343.61112983524947, 343.8916609826643, 343.79536222460797, 343.42021658430195, 342.90858659071966, 342.40283477283475, 342.04532365962086, 341.9454432183466, 342.08069316946137, 342.39560067170953, 342.8346928838355, 343.34249696458386, 343.8675609639535, 344.37451649696135, 344.8320160698784, 345.20871218897605, 345.47325736052574, 345.58515055802394, 345.4672766238686, 345.03336686768296, 344.19715259909015, 342.8723651277134, 341.0461512899859, 338.99932002958, 337.0860958169778, 335.6607031226614, 335.0773664171133, 335.54059103637104, 336.6560057786953, 337.87952030790194, 338.66704428780696, 338.47448738222664, 336.9406292860059, 334.43572981810667, 331.5129188285193, 328.7253261672346, 326.6260816842434, 325.6244928072976, 325.5545772751943, 326.10653040449154, 326.97054751174767, 327.83682391352113, 328.45817071319465, 328.8378621614492, 329.0417882957902, 329.13583915372305, 329.18590477275353, 329.2288719598015, 329.18561459944556, 328.94800334567856, 328.40790885249356, 327.4572017738838, 326.0286998462058, 324.2190091352711, 322.1656827892547, 320.0062739563314, 317.8783357846767, 315.912608665871, 314.21258196511775, 312.8749322910257, 311.99633625220395, 311.67347045726143, 311.94259635710597, 312.5983147718407, 313.3748113638676, 314.00627179558865, 314.22688172940605, 313.8612954284779, 313.0960415589875, 312.2081173878745, 311.47452018207804, 311.17224720853807, 311.48095097103965, 312.190904920751, 312.9950377456858, 313.5862781338581, 313.6575547732818, 313.0053648680698, 311.84047968673144, 310.477239013875, 309.2299826341089, 308.4130503320416, 308.2384926105178, 308.5092028453272, 308.92578513049574, 309.1888435600496, 308.99898222801477, 308.15145243244996, 306.82009428754486, 305.2733951115216, 303.77984222260244, 302.6079229390099, 301.96306264054044, 301.79843895328895, 302.0041675649244, 302.47036416311585, 303.08714443553254, 303.75485178764836, 304.414740496156, 305.01829255555316, 305.5169899603373, 305.8623147050061, 306.0191589951574, 306.00605588079065, 305.8549486230057, 305.597780482902, 305.2664947215798, 304.89339869375027, 304.51225612857206, 304.15719484881515, 303.8623426772498, 303.66182743664626, 303.5748433096576, 303.56084991846865, 303.5643732451473, 303.529939271761, 303.40207398037785, 303.15537605281924, 302.8847349699213, 302.7151129122735, 302.7714720604657, 303.1787745950878, 304.0043706820992, 305.0851624289382, 306.20043992841283, 307.1294932733312, 307.6516125565016, 307.61680776117817, 307.1579684323996, 306.47870400565085, 305.7826239164165, 305.2733376001817, 305.10107898149784, 305.2025799411833, 305.46119684912276, 305.7602860752012, 305.98320398930355, 306.05188966794964, 306.0426130141983, 306.0702266377431, 306.24958314827757, 306.69553515549575, 307.4744293107375, 308.458588431929, 309.4718293786427, 310.33796901045116, 310.8808241869269, 310.98123603403025, 310.74814274327343, 310.3475067725563, 309.94529057977877, 309.70745662284094, 309.75628557656626, 310.039330983473, 310.4604646030027, 310.9235581945971, 311.332483517698, 311.6188998004647, 311.8256161459271, 312.0232291258324, 312.28233531192814, 312.6735312759618, 313.2394315625059, 313.91072260743414, 314.59010881944505, 315.18029460723733, 315.58398437950984, 315.75077790726897, 315.8178564107517, 315.9692964725025, 316.38917467506604, 317.2615676009872, 318.6960809783635, 320.50443711750586, 322.42388747427793, 324.19168350454356, 325.5450766641668, 326.3010463640017, 326.5954838348647, 326.64400826256195, 326.6622388329002, 326.8657947316861, 327.4233808766193, 328.31604511297274, 329.47792101791185, 330.8431421686028, 332.34584214221155, 333.9110983363585, 335.42776343048354, 336.77563392448064, 337.8345063182441, 338.4841771116685, 338.6495927431827, 338.4362994053547, 337.9949932292871, 337.47637034608255, 337.03112688684377, 336.7901370608176, 336.804987389828, 337.1074424738427, 337.7292669128299, 338.70222530675784, 340.0251831716804, 341.56540968799663, 343.1572749521912, 344.6351490607492, 345.83340211015565, 346.6393939661325, 347.15244357135066, 347.5248596377178, 347.9089508771417, 348.45702600153044, 349.27413955742753, 350.2763294299196, 351.3323793387288, 352.31107300357723, 353.0811941441875, 353.54902542894405, 353.77084532088077, 353.84043123169374, 353.85156057307904, 353.8980107567331, 354.0504297794893, 354.2869479787311, 354.5625662769793, 354.83228559675445, 355.05110686057753, 355.17817207649404, 355.18918759464935, 355.0640008507133, 354.7824592803558, 354.3244103192471, 353.690310199244, 352.9630503369504, 352.2461309451571, 351.6430522366547, 351.2573144242342, 351.1617546663604, 351.3065579041947, 351.61124602457227, 351.9953409143287, 352.3783644602995, 352.702581976218, 353.0012324834102, 353.33029843009996, 353.7457622645109, 354.30360643486733, 355.02983057012176, 355.83050302214207, 356.58170932352493, 357.15953500686675, 357.44006560476464, 357.3437661642359, 356.9686197899828, 356.45698910112804, 355.9512367167947, 355.5937252561058, 355.4938447511078, 355.6290948855417, 355.94400275607165, 356.3830954593621, 356.8909000920775, 357.41596464453687, 357.9229206816788, 358.38042066209636, 358.7571170443825, 359.0216622871306, 359.1335553350036, 359.0156810769445, 358.5817708879656, 357.7455561430795, 356.42076821729904, 354.5945540037975, 352.54772246839127, 350.634498095057, 349.2091053677719, 348.62576877051305, 349.08899364128825, 350.20440873422956, 351.42792365749955, 352.21544801926086, 352.0228914276767, 350.48903353280656, 347.98413415229805, 345.06132314569544, 342.27373037254324, 340.17448569238604, 339.1728965475691, 339.10298071164203, 339.65493354095497, 340.5189503918584, 341.38522662070284, 342.00657336040655, 342.3862648501596, 342.59019110571984, 342.6842421428452, 342.73430797729384, 342.77727539401604, 342.7340182547313, 342.4964071903517, 341.95631283178903, 341.00560580995545, 339.577103846174, 337.76741302341196, 335.71408651504754, 333.5546774944594, 331.4267391350258, 329.4610118508063, 327.7609850185845, 326.4233352558248, 325.54473917999184, 325.2218734085503, 325.49099939599324, 326.146717944928, 326.92321469499, 327.55467528581505, 327.7752853570389, 327.40969915306727, 326.6444453373871, 325.75652117825496, 325.02292394392765, 324.7206509026623, 325.0293545622102, 325.73930838830137, 326.54344108616044, 327.1346813610118, 327.20595791808034, 326.55376797461787, 325.38888279598586, 324.025642159573, 322.7783858427683, 321.96145362296033, 321.7868959951825, 322.05760632504627, 322.47418869580736, 322.7372471907217, 322.5473858930454, 321.6998560934428, 320.3684979122135, 318.82179867706526, 317.32824571570615, 316.1563263558442, 315.51146598600616, 315.3468422379936, 315.55257080442675, 316.018767377926, 316.6355476511117, 317.30325503205853, 317.9631437906585, 318.5666959122581, 319.0653933822038, 319.4107181858421, 319.56756252105305, 319.55445943585136, 319.40335219078514, 319.1461840464028, 318.81489826325276, 318.4418021968046, 318.06065958221427, 317.7055982495591, 317.4107460289164, 317.21023075036356, 317.123246602276, 317.10925320622107, 317.11277654206424, 317.0783425896708, 316.9504773289064, 316.703779438943, 316.4331383961804, 316.26351637632484, 316.3198755550827, 316.7271781081605, 317.5527741980263, 318.6335659341954, 319.7488434129449, 320.6778967305516, 321.2000159832928, 321.1652111577375, 320.70637180162305, 320.02710735297865, 319.3310272498337, 318.8217409302177, 318.64948232020066, 318.750983298016, 319.0096002299379, 319.3086894822403, 319.53160742119763, 319.6002931202077, 319.5910164811638, 319.61863011308293, 319.79798662498206, 320.24393862587834, 321.0228327670561, 322.0069918688692, 323.0202327939382, 323.8863724048844, 324.42922756432876, 324.5296394006767, 324.29654610547186, 323.8959101360422, 323.4936939497156, 323.25586000382026, 323.3046889723429, 323.58773439590436, 324.0088680317841, 324.47196163726187, 324.8808869696172, 325.16730325541147, 325.374019598333, 325.57163257135176, 325.830738747438, 326.221934699562, 326.78783497352157, 327.45912600642714, 328.1385122082165, 328.7286979888278, 329.13238775819923, 329.2991812881373, 329.36625979792217, 329.5176998687025, 329.9375780816268, 330.80997101784385, 332.2444844042115, 334.05284055042426, 335.9722909118857, 337.74008694399936, 339.09348010216934, 339.8494497970746, 340.1438872604964, 340.1924116794913, 340.21064224111586, 340.41419813242703, 340.97178427215067, 341.86444850569075, 343.02632441012037, 344.39154556251276, 345.8942455399415, 347.45950173979355, 348.97616684071164, 350.32403734165223, 351.3829097415719, 352.03258053942744, 352.19799617293455, 351.9847028348463, 351.5433966566747, 351.024773769932, 350.57953030613044, 350.33854047495515, 350.3533907987823, 350.6558458781608, 351.27767031363965, 352.250628705768, 353.5735865709963, 355.1138130893808, 356.705678356879, 358.1835524694485, 359.38180552304715, 360.18779738291386, 360.70084699141375, 361.07326306019297, 361.457354300898, 362.00542942517524, 362.8225429794344, 363.8247328491383, 364.8807827545128, 365.8594764157839, 366.62959755317775, 367.0974288355022, 367.31924872589354, 367.3888346360693, 367.39996397774746, 367.446414162646, 367.5988331875502, 367.83535138951436, 368.11096969066, 368.3806890131086, 368.59951027898217, 368.72657549601456, 368.7375910143903, 368.612404269906, 368.33086269835854, 367.87281373554487, 367.2387136134714, 366.51145374898385, 365.7945343551371, 365.1914556449864, 364.80571783158695, 364.71015807359174, 364.8549613120446, 365.15964943358733, 365.5437443248616, 365.92676787250923, 366.2509853900797, 366.5496358987535, 366.87870184661887, 367.29416568176384, 367.85200985227704, 368.578233987031, 369.37890643803735, 370.1301127380918, 370.70793841999034, 370.9884690165291, 370.8921695748974, 370.5170231998584, 370.00539251056875, 369.4996401261848, 369.1421286658634, 369.0422481616513, 369.17749829715575, 369.49240616887414, 369.9314988733039, 370.4393035069426, 370.96436805997604, 371.47132409734314, 371.92882407767064, 372.30552045958586, 372.57006570171603, 372.681958748772, 372.56408448980017, 372.13017429993033, 371.2939595542927, 369.9691716280174, 368.1429574143642, 366.09612587911175, 364.1829015061682, 362.7575087794419, 362.1741721828412, 362.6373970543057, 363.7528121479006, 364.9763270717219, 365.76385143386597, 365.5712948424292, 364.0374369474287, 361.53253756656545, 358.6097265594609, 355.8221337857369, 353.7228891050154, 352.72129995970994, 352.65138412340286, 353.20333695246825, 354.0673538032803, 354.9336300322134, 355.5549767721942, 355.9346682623596, 356.1385945183989, 356.23264555600133, 356.2827113908565, 356.32567880785837, 356.2824216687203, 356.0448106043606, 355.5047162456971, 354.55400922364834, 353.1255072595509, 351.3158164364157, 349.2624899276721, 347.10308090674965, 344.9751425470779, 343.0094152627549, 341.3093884305534, 339.97173866791474, 339.0931425922804, 338.7702768210918, 339.03940280881744, 339.6951213580347, 340.47161810834774, 341.1030786993609, 341.3236887706788, 340.9581025666859, 340.1928487508881, 339.3049245915716, 338.5713273570223, 338.2690543155266, 338.5777579748625, 339.28771180077655, 340.09184449850693, 340.6830847732921, 340.75436133037044, 340.1021713870008, 338.93728620852403, 337.5740455723008, 336.32678925569235, 335.5098570360597, 335.3352994084131, 335.6060097383586, 336.0225921091513, 336.2856506040464, 336.09578930629925, 335.24825950657737, 333.9169013251981, 332.3702020898913, 330.8766491283866, 329.7047297684139, 329.059869398517, 328.8952456504949, 329.10097421696054, 329.56717079052714, 330.18395106380774, 330.8516584448684, 331.51154720358835, 332.1150993252995, 332.613796795334, 332.9591215990242, 333.11596593423985, 333.1028628490024, 332.9517556038708, 332.6945874594042, 332.3633016761617, 331.99020560962333, 331.60906299495275, 331.25400166223454, 330.95914944155305, 330.75863416299325, 330.6716500149345, 330.65765661893636, 330.661179954853, 330.626746002539, 330.49888074184884, 330.2521828519452, 329.9815418092245, 329.8119197893915, 329.8682789681508, 330.27558152120724, 331.1011776110292, 332.18196934713916, 333.29724682582344, 334.2263001433679, 334.7484193960591, 334.71361457047317, 334.25477521434743, 333.57551076570905, 332.8794306625854, 332.3701443430039, 332.197885733032, 332.29938671089735, 332.55800364286773, 332.85709289521077, 333.0800108341945, 333.1486965332121, 333.13941989415906, 333.16703352605595, 333.3463900379239, 333.79234203878383, 334.57123617992397, 335.5553952817014, 336.5686362067408, 337.4347758176666, 337.97763097710356, 338.07804281345926, 337.8449495182744, 337.4443135488725, 337.04209736257746, 336.80426341671307, 336.8530923852622, 337.13613780884407, 337.557271444737, 338.02036505021914, 338.42929038256915, 338.71570666834776, 338.92242301124645, 339.1200359842389, 339.37914216029895, 339.77033811240085, 340.3362383863454, 341.0075294192433, 341.6869156210321, 342.2771014016493, 342.6807911710329, 342.8475847009884, 342.9146632107939, 343.0661032815948, 343.4859814945373, 344.3583744307672, 345.7928878171401, 347.6012439633514, 349.52069432480585, 351.2884903569084, 352.64188351506436, 353.3978532099541, 353.6922906733605, 353.7408150923419, 353.7590456539564, 353.9626015452626, 354.52018768498766, 355.4128519185344, 356.57472782297424, 357.939948975379, 359.4426489528204, 361.0079051526839, 362.52457025361156, 363.872440754559, 364.9313131544821, 365.58098395233685, 365.74639958583873, 365.53310624774184, 365.0918000695598, 364.5731771828063, 364.12793371899534, 363.88694388781295, 363.9017942116358, 364.2042492910127, 364.8260737264928, 365.7990321186252, 367.12198998386015, 368.66221650225293, 370.25408176975975, 371.73195588233716, 372.9302089359418, 373.7362007958116, 374.2492504043118, 374.6216664730892, 375.0057577137904, 375.55383283806236, 376.3709463923153, 377.3731362620127, 378.42918616738103, 379.4078798286474, 380.17800096603855, 380.6458322483629, 380.86765213875634, 380.9372380489357, 380.94836739061856, 380.99481757552223, 381.14723660043126, 381.3837548023998, 381.65937310354866, 381.92909242599933, 382.14791369187316, 382.2749789089039, 382.2859944272765, 382.1608076827882, 381.8792661112363, 381.4212171484186, 380.7871170263419, 380.0598571618522, 379.34293776800433, 378.7398590578536, 378.3541212444554, 378.2585614864626, 378.4033647249188, 378.7080528464651, 379.0921477377427, 379.4751712853931, 379.79938880296515, 380.0980393116396, 380.42710525950446, 380.84256909464824, 381.40041326515944, 382.126637399911, 382.92730985091464, 383.67851615096646, 384.25634183286286, 384.5368724294002, 384.44057298776806, 384.0654266127297, 383.5537959234412, 383.048043539059, 382.69053207873964, 382.59065157452955, 382.72590171003594, 383.0408095817559, 383.47990228618676, 383.98770691982594, 384.5127714728589, 385.0197275102248, 385.4772274905508, 385.8539238724643, 386.1184691145927, 386.2303621616473, 386.11248790267433, 385.6785777128039, 384.842362967166, 383.517575040891, 381.69136082723867, 379.6445292919875, 377.73130491904533, 376.3059121923204, 375.7225755957209, 376.18580046718625, 377.30121556078154, 378.5247304846028, 379.31225484674655, 379.11969825530906, 377.58584036030766, 375.08094097944326, 372.1581299723377, 369.3705371986127, 367.27129251789046, 366.2697033725848, 366.1997875362778, 366.75174036534355, 367.6157572161562, 368.48203344509005, 369.1033801850717, 369.483071675238, 369.68699793127803, 369.781048968881, 369.83111480373634, 369.87408222073816, 369.83082508159976, 369.5932140172394, 369.0531196585753, 368.1024126365258, 366.6739106724277, 364.86421984929194, 362.81089334054803, 360.6514843196253, 358.5235459599536, 356.55781867563087, 354.8577918434299, 353.5201420807919, 352.64154600515815, 352.3186802339701, 352.5878062216962, 353.2435247709137, 354.0200215212269, 354.65148211224, 354.87209218355764, 354.50650597956434, 353.7412521637661, 352.85332800444905, 352.11973076989926, 351.81745772840316, 352.12616138773893, 352.836115213653, 353.6402479113836, 354.231488186169, 354.3027647432476, 353.65057479987837, 352.4856896214018, 351.12244898517895, 349.87519266857066, 349.0582604489382, 348.8837028212916, 349.154413151237, 349.57099552202953, 349.8340540169244, 349.644192719177, 348.79666291945483, 347.46530473807536, 345.9186055027683, 344.4250525412634, 343.25313318129076, 342.6082728113939, 342.443649063372, 342.64937762983794, 343.1155742034048, 343.7323544766856, 344.40006185774655, 345.0599506164666, 345.6635027381777, 346.1622002082122, 346.5075250119023, 346.6643693471178, 346.6512662618802, 346.50015901674846, 346.2429908722816, 345.911705089039, 345.5386090225005, 345.15746640782993, 344.8024050751117, 344.50755285443034, 344.30703757587065, 344.22005342781205, 344.20606003181405, 344.20958336773083, 344.1751494154169, 344.0472841547268, 343.8005862648232, 343.52994522210247, 343.36032320226934, 343.4166823810286, 343.82398493408493, 344.64958102390676, 345.7303727600167, 346.84565023870084, 347.77470355624524, 348.2968228089364, 348.2620179833505, 347.8031786272249, 347.1239141785866, 346.42783407546307, 345.91854775588166, 345.7462891459099, 345.8477901237754, 346.1064070557458, 346.40549630808886, 346.62841424707244, 346.69709994608974, 346.6878233070364, 346.7154369389332, 346.8947934508011, 347.3407454516614, 348.1196395928022, 349.10379869458046, 350.1170396196202, 350.98317923054566, 351.52603438998113, 351.6264462263341, 351.3933529311464, 350.9927169617432, 350.5905007754496, 350.352666829591, 350.40149579815034, 350.68454122174285, 351.1056748576411, 351.5687684631181, 351.97769379544684, 352.264110081187, 352.47082642404615, 352.6684393970182, 352.92754557309746, 353.3187415252785, 353.8846417993664, 354.55593283241205, 355.23531903427676, 355.8255048148224, 356.2291945839106, 356.39598811333116, 356.4630666225855, 356.61450669310295, 357.0343849063129, 357.9067778436451, 359.3412912320149, 361.14964738028294, 363.0690977427957, 364.8368937738998, 366.190286927942, 366.9462566153793, 367.2406940711096, 367.289218486141, 367.30744905148174, 367.51100495814035, 368.06859112567855, 368.9612553878731, 370.1231313070543, 371.4883524455525, 372.9910523656982, 374.5563084617619, 376.07297345577445, 377.42084390170675, 378.47971635352985, 379.1293873652148, 379.2948033861031, 379.0815104470189, 378.6402044741561, 378.1215813937094, 377.6763371318731, 377.43534585494456, 377.45019468963216, 377.7526490027469, 378.3744741611, 379.34743553150287, 380.6703987923359, 382.21063086825706, 383.8024989954932, 385.2803704102716, 386.4786123488195, 387.28458407204334, 387.7976129395655, 388.17001833568787, 388.5541196447121, 389.10223625094017, 389.9194249561795, 390.92169223226074, 391.97778196851976, 392.95643805429285, 393.7264043789163, 394.19395519394095, 394.41548619977704, 394.48492345904896, 394.49619303438163, 394.54322098839987, 394.6966867315212, 394.934283065335, 395.21045613922314, 395.4796521025677, 395.69631710475096, 395.8194759162322, 395.82646779178083, 395.69921060724334, 395.41962223846616, 394.96962056129627, 394.35009934320533, 393.6378559181658, 392.92866351177537, 392.3182953496318, 391.902524657333, 391.7525556889474, 391.84131681242604, 392.11716742419134, 392.5284669206657, 393.02357469827075, 393.5508501534293, 394.05865268256287, 394.4953416820939, 394.8092765484446]
arr_lst = np.array(lst)


class Case1ExampleBot(UTCBot):
    '''
    An example bot for Case 1 of the 2022 UChicago Trading Competition. We recommend that you start
    by reading through this bot and understanding how it works. Then, make a copy of this file and
    start trying to write your own bot!
    '''

    async def handle_round_started(self):
        '''
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        '''
        self.fair_index = 0 # set to 252 * (year_of_simulation - 2021)
        self.rain = []

        self.fairs = {}
        self.order_book = {}
        self.pos = {}
        self.order_ids = {}
        for month in CONTRACTS:
            self.order_ids[month+' bid'] = ''
            self.order_ids[month+' ask'] = ''

            self.fairs[month] = 330

            self.order_book[month] = {
                'Best Bid':{'Price':0,'Quantity':0},
                'Best Ask':{'Price':0,'Quantity':0}}
            
            self.pos[month] = 0

        asyncio.create_task(self.update_quotes())

    def update_fairs(self):
        '''
        You should implement this function to update the fair value of each asset as the
        round progresses.
        '''
        for month in CONTRACTS:
            self.fairs[month] = arr_lst[CONTRACT_TO_DAY[month]]

    async def update_quotes(self):
        '''
        This function updates the quotes at each time step. In this sample implementation we 
        are always quoting symetrically about our predicted fair prices, without consideration
        for our current positions. We don't reccomend that you do this for the actual competition.
        '''
        while True:

            self.update_fairs()

            for contract in CONTRACTS:
                bid_response = await self.modify_order(
                    self.order_ids[contract+' bid'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    ORDER_SIZE,
                    round(self.fairs[contract]-.01*SPREAD,2))

                ask_response = await self.modify_order(
                    self.order_ids[contract+' ask'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    ORDER_SIZE,
                    round(self.fairs[contract]+.01*SPREAD,2))

                assert bid_response.ok
                self.order_ids[contract+' bid'] = bid_response.order_id  
                    
                assert ask_response.ok
                self.order_ids[contract+' ask'] = ask_response.order_id  
            
            await asyncio.sleep(1)

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        This function receives messages from the exchange. You are encouraged to read through
        the documentation for the exachange to understand what types of messages you may receive
        from the exchange and how they may be useful to you.
        
        Note that monthly rainfall predictions are sent through Generic Message.
        '''
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            print('Realized pnl:', update.pnl_msg.realized_pnl)
            print("M2M pnl:", update.pnl_msg.m2m_pnl)

        elif kind == "market_snapshot_msg":
        # Updates your record of the Best Bids and Best Asks in the market
            for contract in CONTRACTS:
                book = update.market_snapshot_msg.books[contract]
                if len(book.bids) != 0:
                    best_bid = book.bids[0]
                    self.order_book[contract]['Best Bid']['Price'] = float(best_bid.px)
                    self.order_book[contract]['Best Bid']['Quantity'] = best_bid.qty

                if len(book.asks) != 0:
                    best_ask = book.asks[0]
                    self.order_book[contract]['Best Ask']['Price'] = float(best_ask.px)
                    self.order_book[contract]['Best Ask']['Quantity'] = best_ask.qty
        
        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.pos[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.pos[fill_msg.asset] -= update.fill_msg.filled_qty

        elif kind == "generic_msg":
            # Saves the predicted rainfall
            try:
                pred = float(update.generic_msg.message)
                self.rain.append(pred)
            # Prints the Risk Limit message
            except ValueError:
                print(update.generic_msg.message)


if __name__ == "__main__":
    start_bot(Case1ExampleBot)