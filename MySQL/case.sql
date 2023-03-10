-- userid 별 총구매액 --
use sqldb;
select userID, sum(price*amount) as '총구매액'
from buytbl
group by userID
order by sum(price*amount) desc;

-- userid 별 총구매액, 이름 출력, 구매 많이 한 순서 -- 
select b.userID, u.name, sum(price*amount) as '총구매액'
from buytbl b
inner join usertbl u on b.userID = u.userID
group by b.userID, u.name
order by sum(price*amount) desc;

-- 전체 회원의 총 구매액 --
select u.userID, u.name, sum(price*amount) as '총구매액'
from buytbl b
right outer join usertbl u on b.userID = u.userID
group by u.userID, u.name
order by sum(price*amount) desc;

-- 구매액 별 고객등급 산정 --
select u.userID, u.name, sum(price*amount) as '총구매액',
case
	when (sum(price*amount) >= 1500) then '최우수고객'
    when (sum(price*amount) >= 1000) then '우수고객'
    when (sum(price*amount) >= 1) then '일반고객'
    else '유령고객'
end as '고객등급'
from buytbl b
right outer join usertbl u on b.userID = u.userID
group by u.userID, u.name
order by sum(price*amount) desc;
