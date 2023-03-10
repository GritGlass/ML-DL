create table indextbl
(first_name varchar(14), last_name varchar(16), hire_date date);
-- employees 테이블에서 데이터 500개 가져오기 --
insert into indextbl
	select first_name, last_name, hire_date
    from employees.employees
    limit 500
    
select * from indextbl where first_name='Mary';

create index idx_indextbl_firstname on indextbl(first_name);