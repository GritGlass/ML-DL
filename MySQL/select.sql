USE mysql;

use employees;
select * from employees;

select * from employees.titles;

-- <실습 1> --

show databases;

use employees;

show table status;

show tables;

describe employees; -- 또는 desc employees;

select first_name, gender from employees;




