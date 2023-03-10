USE sqldb;
create table pivottest
(uName char(3),
season char(2),
amount int);

insert into pivottest values
('김범수','겨울',10),('윤종신','여름',15),('김범수','가을',25),('김범수','봄',3),('김범수','봄',37),
('윤종신','겨울',40),('김범수','여름',14),('김범수','겨울',22),('윤종신','여름',64);

select * from pivottest;

select uNAME,
sum(if(season='봄',amount,0)) as '봄',
sum(if(season='여름',amount,0)) as '여름',
sum(if(season='가을',amount,0)) as '가을',
sum(if(season='겨울',amount,0)) as '겨울',
sum(amount) as '합계' from pivottest group by uName;
