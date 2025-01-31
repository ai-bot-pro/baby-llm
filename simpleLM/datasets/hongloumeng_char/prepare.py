"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    data_url = (
        "https://raw.githubusercontent.com/shjwudp/shu/master/books/%E7%BA%A2%E6%A5%BC%E6%A2%A6.txt"
    )
    with open(input_file_path, "w") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(ll):
    return "".join([itos[i] for i in ll])  # decoder: take a list of integers, output a string


# create the train and test splits
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

# length of dataset in characters: 878,935
# all the unique characters:
# !"()-.:<>?—―‘’“”…─　、。《》〔〕ㄚㄠ一丁七万丈三上下不与丐丑专且世丘丙业丛东丝丞丢两严丧个丫中丰串临丸丹为主丽举乃久么义之乌乍乎乏乐乔乖乘乙乜九乞也习乡书乩买乱乳乾了予争事二于亏云互五井亘些亡亢交亥亦产亩享京亭亮亲亵亸人亿什仁仃仄仅仆仇今介仍从仓仔仕他仗付仙代令以仪们仰仲仵件价任份仿伊伍伏伐休众优伙会伞伟传伤伦伫伯估伴伶伸伺似伽但位低住佐佑体何余佚佛作佞你佣佩佯佳佻使侃侄侈例侍侑供依侠侥侧侪侬侮侯侵便促俄俊俏俐俑俗俘俚保俞俟信俦俨俩俭修俯俱俵俸俺俾倍倏倒倔倕倘候倚借倡倦倩倪倭债值倾偃偄假偈偌偏偕做停健偶偷偿傅傍傧催傲傻像僚僧僭僵僻儒儿允元兄充兆先光克免兑兔党兜兢入全八公六兮兰共关兴兵其具典兹养兼兽冀内冉册再冒冗写军农冠冢冤冥冬冯冰冲决况冶冷冻冽净凄准凉凋凌减凑凛凝几凡凤凫凭凯凰凳凶凸凹出击凿刀刁分切刎刑划列刘则刚创初删判刨利别刮到制刷券刺刻剁剂剃削剌前剎剐剑剔剖剥剧剩剪副割剿劈力劝办功加务劣动助努劫励劲劳劾势勃勇勉勋勒勖勘募勤勾勿匀包匆匈匏匐化北匙匝匠匡匣匪匮匹区医匾匿十千升午卉半卍华协卑卒卓单卖南博卜占卤卦卧卫卯印危即却卵卷卸卿厂厄厅历厉压厌厕厘厚厝原厢厥厦厨厮去县参又叉及友双反发叔取受变叙叛叠口古句另叨叩只叫召叭叮可台叱史右叵叶号司叹叽吁吃各吆合吉吊同名后吏吐向吓吕吗君吝吞吟吣否吧吩含听启吱吴吵吶吸吹吻吾呀呆呈告呕员呛呜呢呦周呱呲味呵呶呷呸呻呼命咂咈和咏咐咒咕咙咚咛咤咧咨咬咭咯咱咳咸咻咽哀品哄哆哇哈哉响哎哑哗哝哟哥哦哧哨哩哪哭哼哽唆唇唏唐唠唤唧唪唬唯唰唱唳唼唾唿啃啄商啊啐啕啖啥啬啰啷啸啻啼啾喀喂喃善喇喉喊喋喏喑喘喜喝喧喳喷喻喽嗄嗅嗐嗓嗔嗜嗟嗡嗣嗤嗦嗳嗷嗽嘁嘈嘉嘎嘘嘟嘱嘲嘴嘻噎噗噙噜噤器噪噫嚎嚏嚷嚼囊囋囔囚四回囟因团园困围囹固国图圃圄圆圈圊土圣在圭地圹场址均坊坍坎坏坐坑块坚坛坞坟坠坡坤坦坳坷垂垄垒垛垢垣垤垦垫埂埃埋城埘埭培基堂堆堑堕堡堤堪堵塌塍塑塔塘塞填塾墀墁境墅墓墙增墟墨墩壁壅壑壤士壬壮声壳壶处备复夏夔夕外夙多夜够大天太夫夭央夯失头夷夸夹夺奁奄奇奈奉奋奎奏契奔奕奖套奚奠奢奥女奴奶奸她奼好如妃妄妆妇妈妍妒妓妖妙妞妤妥妨妩妪妯妹妻妾姆姊始姐姑姓委姜姝姣姥姨姬姻姽姿威娃娄娆娇娈娉娌娑娘娜娟娣娥娲娴娶娼婆婉婕婚婢婪婳婵婶婷婺婿媒媖媚媛媪媳媾嫁嫂嫉嫌嫔嫖嫠嫡嫣嫦嫩嫱嬉嬗嬷孀子孔孕字存孙孝孟季孤学孩孰孳孺孽宁它宅宇守安宋完宏宓宗官定宛宜宝实宠审客宣室宥宦宪宫宰害宴宵家宸容宽宾宿寂寄寅密寇富寐寒寓寝寞察寡寤寥寰寸对寺寻导寿封射将尉尊小少尔尖尘尚尝尤尧尬就尴尸尹尺尼尽尾尿局屁层屃屄居屈屉届屋屎屏屐屑展属屠屣履屯山岁岂岑岔岚岛岩岫岭岳岸峡峥峨峭峰峻崇崔崖崩崽嵇嵌嵘嵩嵬嶂嶒巅巍川州巡巢工左巧巨巫差己已巳巴巷巾市布帅帆师希帏帐帑帔帕帖帘帚帛帜帝带席帮帷常帼帽幄幅幌幔幕幙幛幡幢幪干平年并幸幺幻幼幽广庄庆庇床序庐庑库应底庖店庙庚府庞废度座庭庵庶康庸庾廉廊延廷建开异弃弄弈弊弋式弒弓引弗弘弛弟张弥弦弯弱弹强弼归当录彘彝形彩彪彭彰影彷役彻彼往征径待徇很徊律徐徒得徘御徨循徭微德徽心必忆忌忍忏忒忖志忘忙忝忠忡忤忧快忱忳念忽忿怀态怂怄怅怎怏怒怔怕怖怜思怠怡急性怨怪怯总恁恃恋恍恐恒恕恙恢恣恤恨恩恬恭息恰恳恶恸恹恺恻恼恿悄悉悍悒悔悚悟悠患悦您悫悬悯悲悴悼情惆惊惋惑惓惕惙惚惛惜惟惠惦惧惨惫惬惭惯惰想惶惹惺愁愆愈愍意愕愚感愠愤愦愧愿慈慌慎慑慕慢慧慨慰慵慷憋憎憔憨憩憾懂懈懊懑懒懦懵戈戌戍戎戏成我戒戕或戗战戚戟戥截戮戳戴户戾房所扁扇手才扎扑扒打扔托扛扣执扪扫扬扭扮扯扰扳扶批找承技抄把抑抓抔投抖抗折抚抠抡抢护报披抬抱抵抹押抽抿拂拄担拆拇拈拉拋拌拍拐拒拔拖拗拘拙拚招拜拟拢拣拥拦拧拨择括拭拮拯拱拳拴拷拼拽拾拿持挂指按挑挖挞挟挠挡挣挤挥挦挨挪挫振挲挺挽捂捆捉捍捎捏捐捕捞损捡换捣捧据捱捶捷捺捻掀掂掇授掉掊掌掏掐排掖掘掠探掣接控推掩措掬掯掰掳掷掸掺揆揉揌揎描提插揖握揣揩揪揭揲援揽搀搁搂搅搏搐搓搔搛搜搦搪搬搭搳搴搵携摄摆摇摊摔摘摧摩摸摹撂撅撇撑撒撕撞撤撩播撮撰撵撷撺撼擂擅擉操擎擒擢擤擦擿攀攒攘攥攦攮支收改攻放政敁故效敌敏救敔敕教敛敝敞敢散敦敪敬数敲整敷文斋斑斗料斛斜斝斟斤斧斩断斯新斸方施旁旃旄旅旋旌族旗无既日旦旧旨早旬旱时旷旺昂昆昌明昏易昔星映春昧昨昭是昵昼显晃晋晌晏晒晓晕晖晚晤晦晨普景晰晴晶智晾暂暄暇暑暖暗暮暴暹曦曩曰曲曳更曷曹曼曾替最月有朋服朔朕朗望朝期朦木未末本札朮术朱朴朵机朽杀杂权杆杉杌李杏材村杓杖杜杞束杠条来杨杪杭杯杰杳杷松板极构枇枉析枒枕林枚果枝枢枣枪枫枯枰枳架枷枸柄柏某染柔柘柚柜柢查柩柬柯柱柳柴栅标栉栊栋栎栏树栖栗栩株样核根格栽桀桂桃框案桌桎桐桑桓桔桡档桥桧桨桩桶梁梅梆梏梓梗梢梦梧梨梯械梳梵检棉棋棍棒棔棕棘棚棠森棵棹棺椁椅植椒椿楚楞楠楣楫楷楸楹楼概榄榆榔榛榜榧榭榴榻槁槅槌槎槐槔槛槟槥槽槿樊樗模樨横樯樱樵樽樾橄橇橘橙橱橼檀檐檠欠次欢欣欤欧欲欷欹欺款歇歉歌歍歔止正此步武歧歪歹死殁殂殃殄殆殇殉殊残殒殓殚殡殴段殷殿毁母每毒毓比毕毘毙毛毡毫毯氅氆氇氏民气氤氲水永汀汁求汇汉汕汗汝汞江池污汤汪汰汲沁沈沉沌沏沐沓沙沟没沤沥沦沫河沸油治沼沽沾沿泄泉泊法泛泞泠泡波泣泥注泪泯泰泻泼泽洁洄洇洋洑洒洗洛洞津洪洱洲活洼洽派流浅浆浇浊测济浏浑浓浙浣浥浦浩浪浮浴海浸涂消涉涌涎涕涛涝涟涤润涨涩涮涯液涵涸淅淇淋淌淑淖淘淡淤淫淮深淳混淹添清渊渍渎渐渔渗渚渠渡渣渥温港渴游渺湃湍湖湘湫湮湲湾湿溅溆溉源溘溜溢溪溯溶溺溽滃滇滋滑滓滔滚滞满滥滩滴漂漆漉漏漓演漕漠漫漱潇潘潜潢潦潭潮潸潺潼澄澌澡激濡濣濯灌灞火灭灯灰灵灶灸灼灾灿炉炊炎炒炕炖炙炫炬炭炮炯炳炷炸点炼炽烁烂烈烘烙烛烟烤烦烧烫烬热烹焉焐焕焚焜焦焰然煅煌煎煞煤照煨煮煽熄熊熏熔熙熟熨熬熳燃燎燔燕燥燹爆爇爖爨爪爬爰爱爵父爷爹爻爽片版牌牍牖牙牛牝牟牡牢物牲牵特犀犁犄犒犟犬犯状犹犺狂狄狈狎狐狗狞狠狡独狭狮狰狱狲狷狸狺狼猁猇猖猗猛猜猞猢猩猪猫猬献猴猾猿獐獗獠獭獾玄率玉王玎玑玕玚玛玦玩玫环现玲玳玷玺玻珀珊珍珐珑珖珝珞珠珩班珰球琅理琉琏琐琛琢琥琪琮琳琴琵琶琼瑁瑕瑙瑚瑛瑞瑟瑰瑶璃璎璘璜璧瓜瓟瓠瓢瓣瓤瓦瓮瓯瓶瓷甄甍甘甚甜生甥用甩甬田由甲申电男甸画畀畅界畏畔留畜略畦番畸畿疆疏疑疔疗疙疚疤疮疯疲疵疹疼疾病症痊痌痒痕痘痛痣痨痰痴瘀瘃瘌瘟瘢瘦瘩瘫瘰瘸癖癞癣癫登白百皂的皆皇皎皑皓皤皮皱皴皿盂盄盅盆盈益盏盐监盒盔盖盗盘盛盟盥目直相盹盼盾省眉眊看真眠眦眩眶眷眸眺眼着睁睇睚睛睡睢督睦睨睬睹睿瞅瞋瞎瞒瞟瞥瞧瞪瞬瞭瞳瞻瞽矛矜矢矣知矩矫短矮石矶矼矾码砂砌砍砒研砖砚砣砧砭砰破砸砾硌硝硬确碌碍碎碑碓碗碜碟碣碧碰碴碾磁磊磕磨磬磴磷示礼社祀祇祈祖祗祚祛祝神祟祠祥祧票祭祯祲祷祸祺禀禁禄禅福禧禹离禽禾秀私秃秉秋种科秘租秤秦秧积称移秽秾稀稂程稍税稔稗稚稠稳稻稼稽稿穆穑穗穰穴究穷穸穹空穿窀突窃窄窈窍窑窕窗窘窜窝窟窠窣窥窭窸窿立竖站竞竟章竣童竭端竹竽竿笃笄笆笋笏笑笔笙笛笞笠笤符笨第笺笼筅等筋筏筑筒答策筛筜筝筥筵筷筹签简箍箕算管箦箧箩箪箫箬箭箱箴箸篁篆篇篓篔篙篦篮篱篷篾簇簌簟簠簦簧簪簸簿籁籍籰米类粉粒粗粘粤粥粪粮粱粳粼粽精糊糕糖糙糜糟糠糯系紊素索紧紫紬累絪絮縠縻繁繇纂纕纛纠纡红纣纤约级纨纪纫纬纯纱纲纳纵纶纷纸纹纺纽线练组绅细织终绉绊绍绎经绑绒结绔绕给绛络绝绞统绡绢绣绦继绩绪绫续绮绰绳维绵绸绺绻绽绾绿缀缁缂缄缆缈缉缊缎缓缕编缘缙缚缝缟缠缢缤缥缨缩缪缭缮缯缱缴缶缸缺罄罐网罔罕罗罘罚罢罥罦罩罪罬置署罳罽羁羊羌美羓羔羞羡群羯羲羸羹羼羽翁翅翎翔翘翛翠翡翣翥翦翩翰翻翼耀老考者耆而耍耐耕耗耳耶耸耻耽耿聂聆聊聋职聒联聘聚聪肃肄肆肉肋肌肏肐肓肖肘肚肝肠股肢肤肥肩肯育肴肷肺肿胀胁胃胆背胎胖胜胞胠胡胥胧胫胭胳胶胸能脂脆脉脊脍脏脐脑脓脖脚脧脯脱脸脾腆腊腋腌腐腑腔腕腥腮腰腹腻腼腾腿膀膈膊膏膛膝膨膫膳膺膻臀臂臆臊臜臣自臭至致臻臼臾舀舅舆舌舍舒舔舛舜舞舟舡航舫般舱舵船良艰色艳艺艾节芋芍芎芒芙芜芝芟芥芦芪芬芭芰花芳芷芸芹芽苇苍苏苑苒苓苔苕苗苛苞苟若苦英苹茁茂范茄茅茆茉茎茏茑茔茗茜茞茧茫茯茶茹荆荇草荏荐荑荒荔荚荡荣荤荧荨荫药荳荷荻荼莉莓莠莫莱莲莳获莹莺莼莽菁菂菇菊菌菖菜菡菩菱菲萃萄萌萍萎萝萤营萦萧萨萱萼落葆著葛葡董葩葫葬葭葱葳葵葹葺蒂蒋蒙蒜蒲蒸蒹蒿蓁蓄蓉蓊蓍蓑蓝蓦蓬蓼蔑蔓蔚蔡蔫蔬蔷蔺蔻蔼蔽蕃蕉蕊蕖蕗蕙蕤蕴薄薆薇薋薛薜薨薪薷藉藏藐藓藕藜藤藩藻藿蘅蘑蘩蘸蘼虎虐虑虔虚虞虫虬虱虹虼虽虾虿蚁蚂蚊蚌蚓蚤蚩蚱蛆蛇蛊蛋蛏蛙蛛蛟蛤蛩蛮蛰蛾蜀蜂蜃蜜蜡蜮蜼蝇蝈蝉蝌蝎蝗蝙蝠蝴蝶螂螃融螫螭螯螺蟀蟆蟋蟒蟠蟹蟾蠓蠡蠢蠲蠹血衄衅行衍衔街衙衡衢衣补表衫衬衰衲衷衾衿袂袄袅袋袍袖袜被袭袱袷裀裁裂装裔裕裘裙裢裤裨裱裳裹裾褂褒褓褙褛褡褥褪褫褴褵褶襁襄襜襟西要覆见观规觅视觇览觉觊觌觎觐觑角觚觞解觥触觯言詹誉誊誓謑警譬计订讣认讥讨让讪讫训议讯记讲讳讵讶许讹论讼讽设访证评识诈诉诊诋诌词诎诏诐诓诔试诗诘诙诚诛话诞诟诡询诣该详诧诨诬语诮误诰诱诲诳说诵请诸诺读诼诽课谁调谅谆谇谈谊谋谎谏谐谑谒谓谕谖谗谙谚谛谜谢谣谤谥谦谨谪谬谯谱谲谴谵谶谷豁豆豇豕象豪豫豹貂貉貌贝贞负贡财责贤败账货质贩贪贫贬购贮贯贱贴贵贷贸费贺贻贼贽贾贿赁赃资赈赉赊赋赌赎赏赐赑赔赖赘赚赛赞赠赡赢赤赦赧赫赭走赴赵赶起趁趄超越趋趔趟趣趱足趸趺趾跃跄跋跌跏跐跑跖跛跟跣跤跨跪路跳践跷跸跹跺踉踊踌踏踖踞踟踢踧踩踪踮踱踹蹁蹄蹇蹈蹋蹑蹙蹦蹬蹭蹰蹲蹶蹿躁躇身躬躯躲躺輀车轧轨轩轫转轮软轰轲轳轴轸轻轼载轿较辄辅辆辈辉辍辏辐输辔辕辖辗辘辙辛辜辞辟辣辨辩辫辰辱边达迁迂迄迅过迈迎运近返还这进远违连迟迢迤迥迨迩迫迭述迷迸迹追退送适逃逅逆选逊逍透逐递途逗通逛逝逞速造逡逢逵逶逸逻逼逾遁遂遇遍遏遐道遗遣遥遨遭遮遵遽避邀邂邑邓邙邢那邦邪邬邱邸邻郁郊郎郑郝郡部郭都鄂鄙酉酌配酒酗酡酣酥酪酬酱酴酷酸酹酽酿醁醅醉醋醍醐醑醒醢醪醮醴醺醽醾采释里重野量金釜鉏鉴銮錾鎏针钉钏钓钗钝钟钢钤钥钦钩钮钱钳钵钻钿铁铃铄铅铙铛铜铢铫铭铰铲银铸铺铿销锁锄锅锋错锞锡锢锣锤锥锦锨锭锯锱锲锵锹锻镀镂镇镌镛镜镢镫镯镴镶长门闩闪闭问闯闱闲间闷闸闹闺闻闽阁阃阄阅阆阊阎阏阐阑阔阕阖阗阙阜队阮防阳阴阵阶阻阿陀附际陆陇陈陋陌降限陛陟陡院除陨险陪陵陶陷隅隆隋隍随隐隔隙障隤隧隶隽难雀雁雄雅集雇雉雌雏雕雠雨雪雯雳零雷雹雾需霁霄霆震霉霍霓霖霜霞霭霰露霸霹霾青靖静靛非靠靡面靥革靴靶靸鞋鞍鞓鞘鞫鞭韦韩韫韭音韵韶顑页顶顷顸项顺须顽顾顿颁颂预颅领颇颈颊颏颐频颓颔颖颗题颜额颟颠颡颤颦颧风飐飒飕飖飘飙飞食飧飨餍餐餮饕饥饧饫饬饭饮饯饰饱饲饵饶饺饼饽饿馁馅馆馈馊馋馑馒馔首馗香馥駬騄马驭驮驯驰驱驳驴驸驹驻驼驾驿骂骄骆骇骊验骑骓骗骘骚骛骡骢骤骥骨骯骰骷骸髅髓高髟髯髻鬅鬒鬓鬟鬼魁魂魄魆魇魏魔鮹鱼鲁鲍鲔鲛鲜鲞鲟鲫鲸鳅鳇鳌鳏鳖鳞鳷鶒鸂鸟鸠鸡鸣鸥鸦鸩鸪鸭鸮鸯鸰鸳鸷鸹鸽鸾鸿鹃鹄鹅鹉鹊鹌鹑鹔鹘鹞鹡鹤鹥鹦鹧鹭鹰鹾鹿麀麈麋麒麝麟麦麻麾黄黉黍黎黏黑默黛黜黥黧黹黻黼黾鼋鼎鼐鼒鼓鼠鼻鼾齁齐齑齿龄龙龛龟︰﹐﹒﹔﹕﹗！（），．：；？￥
# vocab size: 4,435
# train has 791,041 tokens
# val has 87,894 tokens
